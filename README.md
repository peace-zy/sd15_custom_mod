# sd15_custom_mod
---

*Latest News* 🔥

- [2025/03] Supports multi-scale image training, uses image cropping information as control conditions, supports long text input, and enables distillation of English models into bilingual Chinese-English models.

---
## Train
```
sh run.sh
```

## Test
```
cd infer_script
sh infer.sh
```

## Other
[Bert tokenizer](https://huggingface.co/google-bert/bert-base-multilingual-uncased/blob/main/config.json)  
[Base Model realisticVisionV51_v51VAE](https://huggingface.co/krnl/realisticVisionV51_v51VAE/tree/main)  
https://modelscope.cn/models/AI-ModelScope/realisticVisionV51_v51VAE
