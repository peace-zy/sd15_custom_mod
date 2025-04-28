# sd15_custom_mod
---

*Latest News* 🔥

- [2025/03] Supports multi-scale image training, uses image cropping information as control conditions, supports long text input, and enables distillation of English models into bilingual Chinese-English models.

---
## Train

1. Download [realisticVisionV51_v51VAE](https://huggingface.co/krnl/realisticVisionV51_v51VAE/tree/main)  
2. Modify unet **config.json** based on **unet_config.json**
```
sh train.sh
```

## Test
```
cd infer_script
sh infer.sh
```

## Other
[Bert tokenizer](https://huggingface.co/google-bert/bert-base-multilingual-uncased/blob/main/config.json)  

https://modelscope.cn/models/AI-ModelScope/realisticVisionV51_v51VAE
