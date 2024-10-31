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
huggingface_hub.login(token="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")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat", "llama2-13b-chat", "longchat-v1.5-7b-32k", "longchat-13b-16k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--version', type=str, default=None, help="FiDComp_version", required=False)

    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
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
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
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
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
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

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", \
                    "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        # datasets = ["hotpotqa"]
        # datasets_path = [("multifieldqa_en", "multifieldqa_en_compressed_context_fidctx_percentage45.json"),
        #                  ("multifieldqa_en", "multifieldqa_en_compressed_context_fidctx_percentage55.json"),
        #                  ("narrativeqa", "narrativeqa_compressed_context_fidctx_percentage10.json"),
        #                  ("narrativeqa", "narrativeqa_compressed_context_fidctx_percentage15.json"),
        #                  ("qasper", "qasper_compressed_context_fidctx_percentage50.json"),
        #                  ("qasper", "qasper_compressed_context_fidctx_percentage70.json")]
        if args.version is not None:
            data_root = f'../R2C/data_compressed/{args.version}'
            # data_root = f'{args.version}'

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
        else:
            print("args.version is not defined, You must be using this code with Longbench default data")
        # datasets = [("hotpotqa", "hotpotqa_compressed_context_fidctx_percentage15.json"),
        #                  ("2wikimqa", "2wikimqa_compressed_context_fidctx_percentage30.json"),
        #                  ("musique", "musique_compressed_context_fidctx_percentage15.json"),
        #                  ("gov_report", "gov_report_compressed_context_fidctx_percentage20.json"),
        #                  ("qmsum", "qmsum_compressed_context_fidctx_percentage15.json"),
        #                  ("multi_news", "multi_news_compressed_context_fidctx_percentage75.json"),
        #                  ("trec", "trec_compressed_context_fidctx_percentage30.json"),
        #                  ("triviaqa", "triviaqa_compressed_context_fidctx_percentage20.json"),
        #                  ("samsum", "samsum_compressed_context_fidctx_percentage25.json"),
        #                  ("passage_count", "passage_count_compressed_context_fidctx_percentage15.json"),
        #                  ("passage_retrieval_en", "passage_retrieval_en_compressed_context_fidctx_percentage15.json"),
        #                  ("lcc", "lcc_compressed_context_fidctx_percentage70.json"),
        #                  ("repobench-p", "repobench-p_compressed_context_fidctx_percentage25.json")]



# datasets = ["hotpotqa", "2wikimqa", "musique", \
#             "gov_report", "qmsum", "multi_news", \
#             "trec", "triviaqa", "samsum", \
#             "passage_count", "passage_retrieval_en", \
#             "lcc", "repobench-p"]

# narrativeqa_compressed_context_fidctx_percentage10.json
# narrativeqa_compressed_context_fidctx_percentage15.json
# multifieldqa_en_compressed_context_fidctx_percentage45.json
# multifieldqa_en_compressed_context_fidctx_percentage55.json
# qasper_compressed_context_fidctx_percentage50.json
# qasper_compressed_context_fidctx_percentage70.json

  # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    # dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    # dataset2prompt = json.load(open("config/dataset2prompt_fid.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt_fid_v4.json", "r"))


    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")


    rank2model = {}
    for rank in range(world_size):
        device = torch.device(f'cuda:{rank}')
        model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
        rank2model[rank] = (model, tokenizer)
    
    for dataset in datasets:
        if isinstance(dataset, tuple):
            is_tuple = True
            dataset, path = dataset ## for compressed data
        else:
            is_tuple = False
            dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
        print(f"> Predicting on {dataset} with model {model_name}")
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            
            if is_tuple:            
                data = json.load(open(path, "r")) ## for compressed data
                if not os.path.exists(f"pred/{model_name}/{args.version}"):
                    os.makedirs(f"pred/{model_name}/{args.version}")
                out_path = f"pred/{model_name}/{args.version}/{dataset}.jsonl" ## for compressed data
            else:
                data = load_dataset('THUDM/LongBench', dataset, split='test')
                if not os.path.exists(f"pred/{model_name}"):
                    os.makedirs(f"pred/{model_name}")
                out_path = f"pred/{model_name}/{dataset}.jsonl"
            
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
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

