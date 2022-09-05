# 使用transformers在nlp两大任务上微调
* 中文文本分类
* 中文文本生成

## 使用模型
* [Bert](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large?text=生活的真谛是%5BMASK%5D%E3%80%82)
* [GPT-2](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall?text=这是很久之前的事情了)

## 准备
### 环境准备
```
conda create -n zh python=3.9
conda activate zh
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```
*可能需要其他库可以pip下载*

### 数据准备 
* [古诗](https://drive.google.com/drive/folders/11sOj1EaWPRIwejmOb0ZPVFN7790mdJxR?usp=sharing)
* [头条新闻](https://drive.google.com/file/d/13DZiVoHeFH8y8TvfAIkdslk6KIuNZKt8/view?usp=sharing)

将数据下载到文件夹 *./MyData* 下，然后修改 *.py*文件中的 *_DATA_PATH* 或者 *_DATA_URL*

## 运行
* 文本分类
```
python BertClassification.py
```

* 文本生成
```commandline
python Seq2SeqForPoterGeneration.py
```

## 实验结果
* 分类acc: 87%  (2个epoch后)