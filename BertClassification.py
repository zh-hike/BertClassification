from transformers import logging
from transformers import AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk, load_metric
import os
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
logging.set_verbosity_error()

_DATA_URL = '../../MyData/nlp'    # 数据所在路径
_model = 'hfl/chinese-bert-wwm-ext'   # 选择模型
_DATA = ['toutiao_cat_data.txt']   # 选择目标文件
test_size = 0.3
NUM_CLASSES = 15


class MyTrainer:
    def __init__(self):
        self.sign = []
        self.tokenizer = AutoTokenizer.from_pretrained(_model)
        self.net = BertForSequenceClassification.from_pretrained(_model, num_labels=NUM_CLASSES)
        try:
            if not os.path.exists('dataset/data'):
                os.mkdir('dataset/data')
            self.data = load_from_disk('dataset/data/')
        except:
            self.data = load_dataset(_DATA_URL, data_files=_DATA, split='train')
            self.data = self.data.train_test_split(test_size=test_size)
            self.data = self.data.map(self.tokenizer_function, batched=True, batch_size=100, remove_columns='text')
            self.data.save_to_disk('dataset/data')
        self.data_collator = DataCollatorWithPadding(self.tokenizer,
                                                     padding=True, )
        self.train_args = TrainingArguments(output_dir='./results',
                                            per_device_train_batch_size=25,
                                            per_device_eval_batch_size=25,
                                            num_train_epochs=15,
                                            evaluation_strategy='epoch',
                                            # eval_steps=10,
                                            dataloader_num_workers=4,
                                            logging_strategy='steps',
                                            logging_steps=3000,
                                            logging_first_step=True,
                                            save_strategy='epoch',
                                            save_total_limit=5,
                                            )
        self.trainer = Trainer(
            self.net,
            self.train_args,
            train_dataset=self.data['train'],
            data_collator=self.data_collator,
            eval_dataset=self.data['test'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,

        )

    def compute_metrics(self, preds):
        acc_metric = load_metric('accuracy')
        f1_metric = load_metric('f1')
        logits, labels = preds
        predict = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=predict, references=labels)
        # f1 = f1_metric.compute(predictions=predict, references=labels)
        metric = {'acc': acc}
        return metric
        # return metric.compute(predictions=predict, references=labels)

    def tokenizer_function(self, example):
        text = example['text']
        x = []
        y = []
        for i in text:
            m = i.split('_!_')
            x.append(m[3])
            label = int(m[1])
            if label in self.sign:
                y.append(self.sign.index(label))
            else:
                y.append(len(self.sign))
                self.sign.append(label)
        token = self.tokenizer(x)
        example['attention_mask'] = token['attention_mask']
        example['input_ids'] = token['input_ids']
        example['token_type_ids'] = token['token_type_ids']
        example['label'] = y
        return example


if __name__ == "__main__":
    trainer = MyTrainer()
    trainer.trainer.train()
