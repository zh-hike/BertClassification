from transformers import logging, DataCollatorWithPadding, TrainingArguments
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer
from datasets import load_dataset
import os
import torch
logging.set_verbosity_error()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICE'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

_DATA_PATH = '../../MyData/nlp'
DATA_FILES = ['poems.txt']
_model = 'pretrainedModel/gpt2-chinese-cluecorpussmall'   # 需要把预训练模型下载到pretrainedModel
test_size = 0.001


class MyTrainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(_model)
        self.net = GPT2LMHeadModel.from_pretrained(_model)
        self.data = load_dataset(_DATA_PATH, data_files=DATA_FILES, split='train')
        self.data = self.data.train_test_split(test_size=test_size)
        self.data = self.data.map(self.tokenizer_function, batched=True, remove_columns='text')
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        self.training_args = TrainingArguments(output_dir='results',
                                               per_device_eval_batch_size=110,
                                               per_device_train_batch_size=110,
                                               num_train_epochs=40,
                                               eval_steps=100,
                                               evaluation_strategy='epoch',
                                               dataloader_num_workers=4,
                                               logging_steps=300,
                                               logging_strategy='steps',
                                               logging_first_step=False,
                                               save_strategy='epoch',
                                               save_total_limit=5,
                                               # disable_tqdm=False,
                                               )
        self.trainer = Trainer(self.net,
                               self.training_args,
                               train_dataset=self.data['train'],
                               data_collator=self.data_collator,
                               tokenizer=self.tokenizer,
                               eval_dataset=self.data['test'],
                               compute_metrics=self.metric,
                               )

    def metric(self, preds):
        start = '春'
        token = self.tokenizer(start, return_tensors='pt', add_special_tokens=False)
        results = self.net.generate(token['input_ids'].cuda(),
                                    do_sample=True,
                                    num_return_sequences=2,
                                    max_length=30,
                                    top_k=4,
                                    use_cache=False,)
        d = self.tokenizer.batch_decode(results, skip_special_tokens=True)
        # print(d)

        return {'result': d}

    def tokenizer_function(self, example):
        d = example['text']
        token = self.tokenizer(d, padding=True, truncation=True, max_length=100)
        example['input_ids'] = token['input_ids']
        example['attention_mask'] = token['attention_mask']
        example['labels'] = token['input_ids']

        return example


if __name__ == "__main__":
    trainer = MyTrainer()
    trainer.trainer.train()
