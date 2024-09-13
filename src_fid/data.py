# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 opt,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 is_eval=False):
        self.data = data
        self.mode = opt.mode
        self.n_contexts = opt.n_contexts
        self.sce_n_contexts = opt.sce_n_contexts
        self.ctx_anno = opt.ctx_anno ## "has_answer, mytho"
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.is_eval = is_eval
        self.t5_tok = T5Tokenizer.from_pretrained('t5-base')
        self.opt = opt
        # self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'answers' in example:
            return random.choice(example['answers'])
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        
        ex_question = example['question']
        if ex_question == '':
            ex_question = "Now, write a one-page summary of the report."
        #     ex_question = "what is the main theme"
        #     ex_question = "what is the title of the text"
        #     ex_question = "what is it about" #################### 현재 1등
        #     ex_question = "What is the main idea?"
        #     ex_question = self.opt.pseudo_question
        #     ex_question = "Can you provide a brief summary of this text?"
        #     ex_question = "what is the purpose of this part"
        #     print(f">> Empty question: {example['id']} >> replace with {ex_question}")
        # print(ex_question)

        ex_question_tokens = self.t5_tok.tokenize(ex_question)
        # print(len(ex_question_tokens))
        if len(ex_question_tokens) > 100:
            print(f">> Long question: {example['id']} >> {len(ex_question_tokens)}")
            ex_question = self.t5_tok.convert_tokens_to_string(ex_question_tokens[-100:])
            # print(ex_question)

        task = example['task']
        question = self.question_prefix + " " + ex_question
        ## I tried question denoising...
        # if self.is_eval:
        #     question = self.question_prefix + " " + ex_question
        # else:
        #     dice = random.random()
        #     if dice < 0.05:
        #         question = self.question_prefix + " "
        #     else:
        #         question = self.question_prefix + " " + ex_question

        target = self.get_target(example)

        if 'ctxs' in example and self.n_contexts is not None:
            text_format_passage = self.passage_prefix + " {}"
            text_format_title_passage = self.title_prefix + " {} " + self.passage_prefix + " {}"

            if example['task'] == 'qa':
                contexts = example['ctxs'][:self.n_contexts]
            else:
                contexts = example['ctxs']
            passages = []
            has_answers = []
            input_ids_ = []

            for c_i, c in enumerate(contexts):
                if c['title'] is None:
                    passages.append(text_format_passage.format(c['text']))
                else:
                    passages.append(text_format_title_passage.format(c['title'], c['text']))
                    
                if c.get('has_answer') is None:
                    has_answer = 0
                else:
                    if self.is_eval:
                        has_answer = c['has_answer'] if c['has_answer'] is not None else 0
                    else:
                        has_answer = c[self.ctx_anno]
                
                has_answers.append(has_answer)                    
                input_ids_.append(c.get('input_ids'))

        else:
            passages = None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'has_answers' : has_answers,
            'input_ids_': input_ids_,
            'task': task,
        }

    def sort_data(self):
        if self.n_contexts is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

    # def convert_examples_to_features(self, tokenizer, question, examples):
    #     question = self.question_prefix + " " + question
    #     text_format = self.title_prefix + " {} " + self.passage_prefix + " {}"
    #     contexts = [text_format.format(example['title'], example['text']) for example in examples]        
    #     text_passages = [question + " " + t for t in contexts]
    #     input_ids, attention_masks = encode_passages([text_passages], tokenizer, 200)
        
    #     return input_ids, attention_masks


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            truncation=True,
            padding='longest',
            return_tensors='pt',
        )
        # p = tokenizer.batch_encode_plus(
        #     text_passages,
        #     max_length=max_length,
        #     truncation=True,
        #     padding='max_length',
        #     return_tensors='pt',
        # )
        
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks.bool()

def process_batch_input_ids(batch_input_ids, max_length):
    ## Padding for those with different lengths
    ## batch_input_ids = list of list of input_ids, e.g., (2, 20, length)
    passage_ids, passage_masks = [], []
    for q_input_ids in batch_input_ids:
        q_passage_ids, q_passage_masks = [], []
        for p_input_ids in q_input_ids:
            len_input_ids = len(p_input_ids)
            if len_input_ids > max_length - 1:
                p_input_ids = p_input_ids[:max_length - 1]
                len_input_ids = max_length - 1

            p_input_ids = p_input_ids + [1] + [0] * (max_length - len_input_ids - 1)
            p_attn_mask = [1] * len_input_ids + [1] + [0] * (max_length - len_input_ids - 1)
            
            q_passage_ids.append(p_input_ids)
            q_passage_masks.append(p_attn_mask)
        passage_ids.append(q_passage_ids)
        passage_masks.append(q_passage_masks)
    passage_ids = torch.cat([torch.LongTensor(passage_ids)], dim=0)
    passage_masks = torch.cat([torch.BoolTensor(passage_masks)], dim=0)

    return passage_ids, passage_masks

class Collator(object):
    def __init__(self, tokenizer, mode, text_maxlength, extra_question):
        self.tokenizer = tokenizer
        self.mode = mode
        self.text_maxlength = text_maxlength
        self.extra_question = extra_question

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        task = [ex['task'] for ex in batch]
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        has_answers = torch.tensor([ex['has_answers'] for ex in batch])
        input_ids_ = [ex['input_ids_'] for ex in batch]

        target = self.tokenizer.batch_encode_plus(
            target,
            padding=True,
            return_tensors='pt'
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example, extra_question=None):
            if example['passages'] is None:
                return [example['question']]
            
            if extra_question == 'embs':
                return [example['question']] + [example['question'] + " " + p for p in example['passages']]
            else:
                return [example['question'] + " " + p for p in example['passages']]

        if input_ids_[0][0] is not None:
            ## input ids are already tokenized and encoded
            passage_ids, passage_masks = process_batch_input_ids(input_ids_, self.text_maxlength)
        else:
            if self.mode == 'pair':
                text_passages = [append_question(example, self.extra_question) for example in batch]
            elif self.mode == 'single':
                text_passages = [[p for p in example['passages']] + [example['question']] for example in batch]

            passage_ids, passage_masks = encode_passages(text_passages,
                                                         self.tokenizer,
                                                         self.text_maxlength)

        question = [example['question'] for example in batch]
        q_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']

        return (index, target_ids, passage_ids, passage_masks, q_tokens, has_answers, task)


def load_data(data_path=None, global_rank=-1, world_size=-1, n_qas=None):
    assert data_path
    # if data_path.endswith('.jsonl'):
    #     data = open(data_path, 'r')
    # elif data_path.endswith('.json'):
    #     with open(data_path, 'r') as fin:
    #         data = json.load(fin)
    with open(data_path, 'r') as fin:
        data = json.load(fin)    
    if n_qas is not None:
        data = data[:n_qas]

    examples = []
    for k, example in enumerate(tqdm(data, desc="> Loading data")):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if example.get('id') is None:
            example['id'] = k

        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples