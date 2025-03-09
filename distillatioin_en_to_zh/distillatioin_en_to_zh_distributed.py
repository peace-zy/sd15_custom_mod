## v6
from ast import mod
import dis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, CLIPTokenizer, CLIPTextModel
from transformers import Trainer, TrainingArguments, PreTrainedModel, HfArgumentParser
from transformers.utils import is_peft_available, is_safetensors_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME, logging
import datasets
from dataclasses import dataclass, field
import re
import json
import os
import torch.distributed as dist
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if is_safetensors_available():
    import safetensors.torch
if is_peft_available():
    from peft import PeftModel

TRAINING_ARGS_NAME = "training_args.bin"

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    teacher_model_name_or_path: str = field(
        default="realisticVisionV51_v51VAE",
        metadata={"help": "The teacher model checkpoint for weights initialization."}
    )
    teacher_tokenizer_subfolder: str = field(
        default="tokenizer",
        metadata={"help": "The subfolder of the teacher tokenizer."}
    )
    teacher_text_encoder_subfolder: str = field(
        default="text_encoder",
        metadata={"help": "The subfolder of the teacher text_encoder."}
    )
    student_model_name_or_path: str = field(
        default="realisticVisionV51_v51VAE",
        metadata={"help": "The student model checkpoint for weights initialization."}
    )
    student_tokenizer_subfolder: str = field(
        default="merged_clip_tokenizer_only_zh_word",
        metadata={"help": "The subfolder of the student tokenizer."}
    )
    student_text_encoder_subfolder: str = field(
        default="merged_text_encoder_only_zh_word",
        metadata={"help": "The subfolder of the student text_encoder."}
    )
    student_tokenizer_type: str = field(
        default="CLIPTokenizer", # [BertTokenizer or CLIPTokenizer]
        metadata={"help": "The type of the student tokenizer."}
    )
    student_fix_clip_embedding_with_teacher: bool = field(
        default=True,
        metadata={"help": "Whether to fix the embedding layer of the student model."}
    )
    student_fix_clip_embedding_name: str = field(
        default="text_encoder.text_model.embeddings.token_embedding.weight",
        metadata={"help": "The name of the embedding layer to fix."}
    )
    student_freeze_clip_backbone: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the backbone of the student model."}
    )
    student_add_decoder: bool = field(
        default=True,
        metadata={"help": "Whether to add a decoder to the student model."}
    )
    student_add_decoder_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the decoder."}
    )
    student_add_decoder_nhead: int = field(
        default=8,
        metadata={"help": "The head size of the decoder."}
    )
    distill_intermediate_layers: bool = field(
        default=False,
        metadata={"help": "Whether to distill the intermediate layers."}
    )
    distill_layer_index: int = field(
        default=-1,
        metadata={"help": "The index of the layer to distill."}
    )

@dataclass
class DistillationLossArguments:
    """
    Arguments pertaining to the loss function.
    """
    loss_type: List[str] = field(
        default_factory=lambda: "MSELoss",
        metadata={"help": "The type of the loss function. Should be in [MSELoss, SmoothL1Loss, CosineEmbeddingLoss]"}
    )
    loss_weight: List[float] = field(
        default_factory=lambda: 1.0,
        metadata={"help": "The weight of the loss."}
    )
    translate_loss_weight: float = field(
        default=1.0,
        metadata={"help": "The weight of the translation loss."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: str = field(
        default='dataset/translation2019zh/translation2019zh_train_clean.jsonl',
        metadata={"help": "The input training data file (a jsonlines file)."}
    )
    test_data_file: str = field(
        default='dataset/translation2019zh/translation2019zh_valid_clean.jsonl',
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a jsonlines file)."}
    )

def contains_chinese(text):
    # 使用正则表达式检查是否存在中文字符
    return re.search(r'[\u4e00-\u9fff]', text) is not None

class ZhEnDataset(Dataset):
    def __init__(self, data_file, teacher, student):
        self.data_file = data_file
        self.teacher = teacher
        self.student = student
        self.max_length = 77
        self.raw_data = self.load_data()
        self.num = len(self.raw_data)

    def __len__(self):
        return len(self.raw_data)

    def str_replace(self, data):
        chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”']
        englishTab = [':', ';', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"']
        for index in range(len(chinaTab)):
            if chinaTab[index] in data:
                data = data.replace(chinaTab[index], englishTab[index])
        data = re.sub(r"([,])\1+", ",", data)
        return data

    def preprocess_function(self, example):
        teacher_en_tokens = self.teacher.tokenizer(
            example['en'], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors='pt'
            )['input_ids']

        student_zh_tokens = self.student.tokenizer(
                        example['zh'],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )['input_ids']

        student_en_tokens = self.student.tokenizer(
                        example['en'],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )['input_ids']
        """
        t = self.teacher.tokenizer(
                example['en'], truncation=True,
                max_length=self.max_length, return_tensors='pt'
                )['input_ids']
        s = self.student.tokenizer(
                example['en'],
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )['input_ids']
        """


        return {
            "teacher_en_tokens": teacher_en_tokens,
            "student_zh_tokens": student_zh_tokens,
            "student_en_tokens": student_en_tokens,
        }

    def load_json(self, line):
        fields = json.loads(line.strip())
        zh, en = fields['chinese'], fields['english']
        """
        en = self.str_replace(en)
        zh = re.sub(r'[^\w\s]', ',', zh)
        zh = re.sub(r"([,])\1+", ",", zh)
        """
        return {'en': en, 'zh': zh}

    def load_data(self):
        raw_data = []
        with open(self.data_file, 'r', errors='ignore', encoding="utf8") as f:
            for line in tqdm(f.readlines(), desc="Loading data"):
                try:
                    json_data = self.load_json(line)
                    if contains_chinese(json_data["en"]):
                        print(f"en contains zh {line}")
                        continue
                    raw_data.append(json_data)
                except Exception as e:
                    print(e)
                    continue
        return raw_data

    def __getitem__(self, idx):
        return self.preprocess_function(self.raw_data[idx])

class CustomDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, add_decoder_nhead, add_decoder_layers):
        super(CustomDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=add_decoder_nhead, batch_first=True),
            num_layers=add_decoder_layers
        )
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        decoder_output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.decoder_output_layer(decoder_output)
        return output

class StudentModel(nn.Module):
    def __init__(self, model_path, tokenizer_subfolder="merged_clip_tokenizer_only_zh_word",
                 text_encoder_subfolder="merged_text_encoder_only_zh_word", tokenizer_type="CLIPTokenizer",
                 freeze_clip_backbone=True, add_decoder=True, add_decoder_layers=2,
                 add_decoder_nhead=8, max_length=77):
        super(StudentModel, self).__init__()
        self.model_path = model_path
        self.max_length = max_length
        self.add_decoder = add_decoder
        if tokenizer_type == "CLIPTokenizer":
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder=tokenizer_subfolder)
            self.text_encoder = CLIPTextModel.from_pretrained(self.model_path, subfolder=text_encoder_subfolder, ignore_mismatched_sizes=True)
            if freeze_clip_backbone:
                for name, param in self.text_encoder.named_parameters():
                    if not name.startswith("text_model.embeddings.token_embedding.weight"):
                        param.requires_grad = False
                #self.text_encoder.text_model.encoder.layers[-1].mlp.fc2.requires_grad = True
            if self.add_decoder:
                # 添加中文翻译
                self.translation = CustomDecoder(
                    hidden_size=self.text_encoder.config.hidden_size,
                    vocab_size=self.tokenizer.vocab_size,
                    add_decoder_nhead=add_decoder_nhead,
                    add_decoder_layers=add_decoder_layers
                )

        elif tokenizer_type == "BertTokenizer":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path, subfolder=tokenizer_subfolder)
            self.text_encoder = BertModel.from_pretrained(self.model_path, subfolder=text_encoder_subfolder)
        else:
            raise ValueError(f"Invalid tokenizer type {tokenizer_type}, should be one of [CLIPTokenizer, BertTokenizer]")

    def forward(self, memory=None, tgt=None, tgt_mask=None):
        if self.add_decoder:
            # 使用中文hidden_state作为解码器的输入
            #print(f"zh_hidden_states.shape={memory.shape}, en_input_ids.shape={tgt.shape}, {tgt}")
            #print(f"tgt_mask={tgt_mask}, tgt_mask.shape={tgt_mask.shape}")
            #print(f"zh_hidden_states.shape={memory.shape}, en_input_ids.shape={tgt.shape}")
            #print(f"tgt_mask.shape={tgt_mask.shape}")
            #zh_hidden_states = zh_hidden_states.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
            #en_input_ids = en_input_ids.permute(1, 0)  # (seq_len, batch_size)
            #en_hidden_states = self.text_encoder(en_input_ids)[0].permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
            #decoder_output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            #print(f"decoder_output.shape={decoder_output.shape}")
            #decoder_output = decoder_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
            #logits = self.decoder_output_layer(decoder_output)
            logits = self.translation(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            return logits
        else:
            pass
        #output_hidden_states=True)["hidden_states"]
        #'last_hidden_state', 'pooler_output', 'hidden_states'
class TeacherModel(nn.Module):
    def __init__(self, model_path, tokenizer_subfolder="tokenizer",
                 text_encoder_subfolder="text_encoder", max_length=77):
        super(TeacherModel, self).__init__()
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder=tokenizer_subfolder)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_path, subfolder=text_encoder_subfolder)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

class DistillationModel(nn.Module):
    def __init__(self, teacher, student, model_args):
        super(DistillationModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher = teacher
        self.teacher.eval()
        self.student = student
        self.model_args = model_args
        self.fixed_weight_name = self.model_args.student_fix_clip_embedding_name
        if self.model_args.student_fix_clip_embedding_with_teacher:
            self.fixed_teacher_weight_name = f"teacher.{self.fixed_weight_name}"
            self.fixed_student_weight_name = f"student.{self.fixed_weight_name}"
            #print(self.fixed_teacher_weight_name, self.fixed_student_weight_name)
            self.fixed_teacher_weight = eval(f"self.{self.fixed_teacher_weight_name}").cpu().detach().clone()
            eval(f"self.{self.fixed_student_weight_name}").requires_grad = False
            self.old_num = self.fixed_teacher_weight.shape[0]
            eval(f"self.{self.fixed_student_weight_name}")[:self.old_num, :] = self.fixed_teacher_weight
            eval(f"self.{self.fixed_student_weight_name}").requires_grad = True

    def forward(self, x=None, return_loss=True, **kwargs):
        if "student_zh_tokens" in kwargs:
            student_zh_tokens = kwargs["student_zh_tokens"].to(self.device)
        else:
            student_zh_tokens = x["student_zh_tokens"].to(self.device)
        if "student_en_tokens" in kwargs:
            student_en_tokens = kwargs["student_en_tokens"].to(self.device)
        else:
            student_en_tokens = x["student_en_tokens"].to(self.device)
        student_zh_preds = self.student.text_encoder(student_zh_tokens, output_hidden_states=True)
        student_en_preds = self.student.text_encoder(student_en_tokens, output_hidden_states=True)

        if "teacher_en_tokens" in kwargs:
            teacher_en_tokens = kwargs["teacher_en_tokens"].to(self.device)
        else:
            teacher_en_tokens = x["teacher_en_tokens"].to(self.device)

        with torch.no_grad():
            teacher_en_preds = self.teacher.text_encoder(teacher_en_tokens, output_hidden_states=True)
        decoder_logits = None

        if self.student.add_decoder:
            tgt = student_en_preds["last_hidden_state"]
            memory = student_zh_preds["last_hidden_state"]
            tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1)), diagonal=0) == 0
            tgt_mask = tgt_mask.to(self.device)
            # 使用中文hidden_state作为解码器的输入
            decoder_logits = self.student(memory, tgt, tgt_mask)

        return {
            "student_zh_preds": student_zh_preds,
            "student_en_preds": student_en_preds,
            "teacher_en_preds": teacher_en_preds,
            "decoder_logits": decoder_logits,
            "student_en_tokens": student_en_tokens,
            "teacher_en_tokens": teacher_en_tokens,
        }

class DistillationTrainer(Trainer):
    # ref:https://huggingface.co/docs/transformers/v4.42.0/en/tasks/knowledge_distillation_for_image_classification#knowledge-distillation-for-computer-vision
    def __init__(self, model_args=None, distill_loss_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_args = model_args
        self.distill_loss_args = distill_loss_args

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.model_args.student_fix_clip_embedding_with_teacher:
            if hasattr(model, "module"):
                fixed_student_weight_name = f"model.module.{model.module.fixed_student_weight_name}"
                #fixed_teacher_weight_name = f"model.module.{model.module.fixed_teacher_weight_name}"
                fixed_teacher_weight = model.module.fixed_teacher_weight
            else:
                fixed_student_weight_name = f"model.{model.fixed_student_weight_name}"
                #fixed_teacher_weight_name = f"model.{model.fixed_teacher_weight_name}"
                fixed_teacher_weight = model.fixed_teacher_weight
            eval(fixed_student_weight_name).requires_grad = False
            old_num = fixed_teacher_weight.shape[0]
            eval(f"{fixed_student_weight_name}")[:old_num, :] = fixed_teacher_weight
            eval(fixed_student_weight_name).requires_grad = True

        preds = model(inputs)
        teacher_en_preds = preds["teacher_en_preds"]
        student_en_preds = preds["student_en_preds"]
        student_zh_preds = preds["student_zh_preds"]

        #s_merge = torch.cat([student_en_preds, student_zh_preds])
        #t_merge = torch.cat([teacher_en_preds] * 2)
        def run_compute_loss(s_merge, t_merge):
            loss_dict = {}
            loss = 0
            for loss_type, loss_weight in zip(self.distill_loss_args.loss_type, self.distill_loss_args.loss_weight):
                if loss_type == "MSELoss":
                    mse_loss = F.mse_loss(s_merge.float(), t_merge.float(), reduction='mean')
                    loss += mse_loss * loss_weight
                    loss_dict[loss_type] = f"{mse_loss * loss_weight}"
                elif loss_type == "SmoothL1Loss":
                    smooth_l1_loss = F.smooth_l1_loss(s_merge.float(), t_merge.float(), reduction='mean')
                    loss += smooth_l1_loss * loss_weight
                    loss_dict[loss_type] = f"{smooth_l1_loss * loss_weight}"
                elif loss_type == "CosineEmbeddingLoss":
                    dim = s_merge.size(-1)
                    s_merge = s_merge.view(-1, dim)
                    t_merge = t_merge.view(-1, dim)
                    target = s_merge.new(s_merge.size(0)).fill_(1)
                    cosine_loss = F.cosine_embedding_loss(s_merge[:, 0].unsqueeze(1).float(), t_merge[:, 0].unsqueeze(1).float(), target, reduction="mean")
                    loss += cosine_loss * loss_weight
                    loss_dict[loss_type] = f"{cosine_loss * loss_weight}"
                else:
                    raise ValueError(f"Invalid loss type {loss_type}, should be one of [MSELoss, SmoothL1Loss, CosineEmbeddingLoss]")
            return loss, loss_dict
        total_loss = 0
        losses = {
            "Learning": {},
            "Watching": {}
        }
        last_hidden_state_loss, last_hidden_state_loss_dict = run_compute_loss(student_zh_preds["last_hidden_state"], teacher_en_preds["last_hidden_state"])
        losses["Watching"]["last_hidden_state"] = last_hidden_state_loss_dict
        for layer_idx, s_merge in enumerate(student_zh_preds["hidden_states"]):
            t_merge = teacher_en_preds["hidden_states"][layer_idx]
            loss, loss_dict = run_compute_loss(s_merge, t_merge)
            if self.model_args.distill_intermediate_layers:
                total_loss += loss
                losses["Learning"][f"hidden_states_{layer_idx}"] = loss_dict
            else:
                losses["Watching"][f"hidden_states_{layer_idx}"] = loss_dict

        if self.model_args.distill_layer_index == -1:
            total_loss += last_hidden_state_loss
            losses["Learning"]["last_hidden_state"] = last_hidden_state_loss_dict
        else:
            if self.model_args.distill_layer_index in range(len(student_zh_preds["hidden_states"])):
                loss, loss_dict = run_compute_loss(student_zh_preds["hidden_states"][self.model_args.distill_layer_index],
                                                   teacher_en_preds["hidden_states"][self.model_args.distill_layer_index])
                total_loss += loss
                losses["Learning"][f"hidden_states_{self.model_args.distill_layer_index}"] = loss_dict
            else:
                raise ValueError(f"Invalid distill_layer_index {self.model_args.distill_layer_index}, "
                                 "should be in range({len(student_zh_preds['hidden_states'])})")

        if self.model_args.student_add_decoder:
            #student_en_tokens = preds["student_en_tokens"]
            #student_en_tokens = student_en_tokens.view(student_en_tokens.size(0), -1)
            teacher_en_tokens = preds["teacher_en_tokens"]
            teacher_en_tokens = teacher_en_tokens.view(teacher_en_tokens.size(0), -1)
            decoder_logits = preds["decoder_logits"]
            #translation_loss = F.cross_entropy(decoder_logits.view(-1, decoder_logits.size(-1)), student_en_tokens.view(-1), ignore_index=self.model.student.tokenizer.eos_token_id)
            translation_loss = F.cross_entropy(decoder_logits.view(-1, decoder_logits.size(-1)).float(), teacher_en_tokens.view(-1).float())
            total_loss += translation_loss * self.distill_loss_args.translate_loss_weight
            losses["Translation_Loss"] = f"{translation_loss * self.distill_loss_args.translate_loss_weight}"

        # 获取当前的步数
        current_step = self.state.global_step
        logging_steps = self.args.logging_steps
        #if current_step % logging_steps == 0 and dist.get_rank() == 1:
        if current_step % logging_steps == 0:
            extra_info = "Distill intermediate layers" if self.model_args.distill_intermediate_layers else ""
            extra_info += "Distill last_hidden_state" if self.model_args.distill_layer_index == -1 else ""
            extra_info += (
                f"Distill hidden_states_{self.model_args.distill_layer_index}"
                if self.model_args.distill_layer_index in range(len(student_zh_preds["hidden_states"]))
                else ""
            )
            info = {
                "Distill param": extra_info,
                "Loss": losses
            }
            #print(f"\nStep: {current_step}\n{json.dumps(info, ensure_ascii=False, indent=4)}")
            #print(f"\n->Step: {current_step}\n->Distill param: {extra_info}\n"
            #      f"->Loss: {json.dumps(losses, ensure_ascii=False, indent=4)}\n")

        total_loss = total_loss / self.args.gradient_accumulation_steps
        return (total_loss, (s_merge, t_merge)) if return_outputs else total_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model = self.model.student
        state_dict = model.text_encoder.state_dict()
        if not isinstance(self.model, supported_classes):
            if isinstance(self.accelerator.unwrap_model(model.text_encoder), supported_classes):
                self.accelerator.unwrap_model(model.text_encoder).save_pretrained(
                    os.path.join(output_dir, 'text_encoder'), state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(os.path.join(output_dir, 'text_encoder'), SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(os.path.join(output_dir, 'text_encoder'), WEIGHTS_NAME))
        else:
            model.text_encoder.save_pretrained(
                os.path.join(output_dir, 'text_encoder'), state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.model_args.student_add_decoder:
            tranlation_path = os.path.join(output_dir, 'translation')
            os.makedirs(tranlation_path, exist_ok=True)
            state_dict = model.translation.state_dict()
            if self.args.save_safetensors:
                safetensors.torch.save_file(
                        state_dict, os.path.join(tranlation_path, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
            else:
                torch.save(state_dict, os.path.join(tranlation_path, WEIGHTS_NAME))

        if model.tokenizer is not None:
            tokenizer_path = os.path.join(output_dir, 'tokenizer')
            os.makedirs(tokenizer_path, exist_ok=True)
            model.tokenizer.save_pretrained(tokenizer_path)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DistillationLossArguments))
    model_args, data_args, training_args, distill_loss_args = parser.parse_args_into_dataclasses()
    assert distill_loss_args.loss_weight is None or \
            len(distill_loss_args.loss_weight) == len(distill_loss_args.loss_type), \
            "The number of loss weights should be equal to the number of loss types."
    if model_args.distill_layer_index != -1 and model_args.distill_intermediate_layers:
        raise ValueError("distill_layer_index and distill_intermediate_layers cannot be set at the same time.")

    teacher = TeacherModel(
        model_path=model_args.teacher_model_name_or_path,
        tokenizer_subfolder=model_args.teacher_tokenizer_subfolder,
        text_encoder_subfolder=model_args.teacher_text_encoder_subfolder
    )

    student = StudentModel(
        model_path=model_args.student_model_name_or_path,
        tokenizer_subfolder=model_args.student_tokenizer_subfolder,
        text_encoder_subfolder=model_args.student_text_encoder_subfolder,
        tokenizer_type=model_args.student_tokenizer_type,
        freeze_clip_backbone=model_args.student_freeze_clip_backbone,
        add_decoder=model_args.student_add_decoder,
        add_decoder_layers=model_args.student_add_decoder_layers,
        add_decoder_nhead=model_args.student_add_decoder_nhead
    )
    """
    if dist.get_rank() == 1:
        print("[Init student model sucess!]")
    """
    print("[Init student model sucess!]")


    distillation_model = DistillationModel(
        teacher=teacher,
        student=student,
        model_args=model_args
    ).to(device)
    """
    if dist.get_rank() == 1:
        print("[Init distillation_model sucess!]")
    """
    print("[Init distillation_model sucess!]")

    train_dataset = ZhEnDataset(
        data_file=data_args.train_data_file,
        teacher=distillation_model.teacher,
        student=distillation_model.student
    )
    test_dataset = ZhEnDataset(
        data_file=data_args.test_data_file,
        teacher=distillation_model.teacher,
        student=distillation_model.student
    )

    trainer = DistillationTrainer(
        model=distillation_model,
        model_args=model_args,
        distill_loss_args=distill_loss_args,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #compute_metrics=compute_metrics
    )

    # print trainable parameters
    #if dist.get_rank() == 1:
    if True:
        print("rank 1!!!!!!")
        print("[Load data sucess!]")
        print(f"model_args={model_args}, data_args={data_args}, training_args={training_args}")
        print("Trainable parameters:")
        for name, param in distillation_model.named_parameters():
            if param.requires_grad:
                logger.info(name)
                print(name)
        print(f"Train data size: {len(train_dataset)}")
        print(f"Test data size: {len(test_dataset)}")

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

if __name__ == '__main__':
    main()
