# Download dataset
```
url in dataset.txt
mkdir dataset && cd dataset
wget -O dataset/ https://storage.googleapis.com/nlp_chinese_corpus/translation2019zh.zip
unzip -X translation2019zh.zip
```
# Filter data
```
python filter_data.py
```
# Merget token
1. Copy **text_encoder** folder of model to **text_encoder_bert** and modify **config.json** based on text_encoder_bert_config.json  
2. Copy **tokenizer_bert** folder to model
```
python merge_token.py
```
# Train
```
sh run.sh
sh run_multi_node.sh
```
