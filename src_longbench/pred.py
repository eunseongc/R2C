import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

import huggingface_hub
huggingface_hub.login(token="") ## Input your token here

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {compressed_prompt}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {compressed_prompt}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{compressed_prompt}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{compressed_prompt}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{compressed_prompt}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{compressed_prompt}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{compressed_prompt}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{compressed_prompt}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{compressed_prompt}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{compressed_prompt}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{compressed_prompt}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{compressed_prompt}",
    "lcc": "Please complete the code given below. \n{compressed_prompt}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{compressed_prompt}Next line of code:\n"
}

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lcc": 64,
    "repobench-p": 64
}

MODEL2PATH = {
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "longchat-13b-16k": "lmsys/longchat-13b-16k",
}

MODEL2MAXLEN = {
    "llama2-7b-chat": 3500,
    "llama2-13b-chat": 3500,
    "longchat-13b-16k": 15500,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat", "llama2-13b-chat", "longchat-v1.5-7b-32k", "longchat-13b-16k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--version', type=str, default=None, help="FiDComp_version", required=False)

    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"

    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(data, max_length, max_gen, prompt_format, dataset, device, model_name, model, tokenizer, out_path):
    for json_obj in tqdm(data, dynamic_ncols=True):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
            
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            # json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": '', "length": 1}, f, ensure_ascii=False)            
            
            f.write('\n')

    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "llama2" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        # replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = MODEL2MAXLEN[model_name]

    data_root = f'data_compressed/{args.version}'

    all_file_list = os.listdir(data_root)
    datasets = []
    for file_name in all_file_list:
        file_name_splitted = file_name.split('_')
        longbench_idx = file_name_splitted.index('longbench')
        fid_idx = file_name_splitted.index('fid')
        dataname = '_'.join(file_name_splitted[longbench_idx+1:fid_idx])
        datasets.append((dataname, os.path.join(data_root, file_name)))
    dataset_ordered = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p", "result"]
    datasets = sorted(datasets, key=lambda x: dataset_ordered.index(x[0]))

    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")

    rank2model = {}
    for rank in range(world_size):
        device = torch.device(f'cuda:{rank}')
        model, tokenizer = load_model_and_tokenizer(MODEL2PATH[model_name], model_name, device)
        rank2model[rank] = (model, tokenizer)
    
    for dataset in datasets:
        dataset, path = dataset ## for compressed data
        print(f"> Predicting on {dataset} with model {model_name}")

        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        
        data = json.load(open(path, "r")) ## for compressed data
        if not os.path.exists(f"pred/{model_name}/{args.version}"):
            os.makedirs(f"pred/{model_name}/{args.version}")
        out_path = f"pred/{model_name}/{args.version}/{dataset}.jsonl" ## for compressed data

        prompt_format = DATASET2PROMPT[dataset]
        max_gen = DATASET2MAXLEN[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        for rank in range(world_size):
            device = torch.device(f'cuda:{rank}')
            model, tokenizer = rank2model[rank]
            processes = []
            p = mp.Process(target=get_pred, args=(data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model, tokenizer, out_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

