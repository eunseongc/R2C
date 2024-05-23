import os
import sys
import json
import pickle
import logging
import argparse
import tiktoken

import numpy as np

from time import time
from xopen import xopen
from copy import deepcopy
from tqdm import tqdm
from src.compress_utils import *
from transformers import T5Tokenizer
from IPython import embed


INSTRUCTION = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
QUESTION_TEMPLATE = "Question: {}\nAnswer:"

_dict = {
    "narrativeqa": "{compressed_prompt}\n\n{question}",
    "qasper": "{compressed_prompt}\n\n {question}",
    "multifieldqa_en": "\n\n{compressed_prompt}\n\n{question}",
    "hotpotqa": "{compressed_prompt}\n\n{question}",
    "2wikimqa": "{compressed_prompt}\n\n{question}",
    "musique": "{compressed_prompt}\n\n{question}",
    "gov_report": "{compressed_prompt}\n\n{question}",
    "qmsum": "{compressed_prompt}\n\n{question}",
    "multi_news": "{compressed_prompt}\n\n{question}",
    "trec": "{compressed_prompt}\n{question}",
    "triviaqa": "{compressed_prompt}\n\n{question}",
    "samsum": "{compressed_prompt}\n\n{question}",
    "passage_count": "{compressed_prompt}\n\n{question}",
    "passage_retrieval_en": "{compressed_prompt}\n\n{question}",
    "lcc": "{compressed_prompt}{question}",
    "repobench-p": "{compressed_prompt}{question}"
}


_dict_wo_instruction = {
    "narrativeqa": "{compressed_prompt}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {question}",
    "qasper": "{compressed_prompt}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {question}",
    "multifieldqa_en": "{compressed_prompt}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {question}",
    "hotpotqa": "{compressed_prompt}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}",
    "2wikimqa": "{compressed_prompt}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}",
    "musique": "{compressed_prompt}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}",
    "gov_report": "{compressed_prompt}",
    "qmsum": "{compressed_prompt}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {question}",
    "multi_news": "{compressed_prompt}",
    "trec": "{compressed_prompt}\n{question}",
    "triviaqa": "{compressed_prompt}\n\n{question}",
    "samsum": "{compressed_prompt}\n\n{question}",
    "passage_count": "{compressed_prompt}",
    "passage_retrieval_en": "{compressed_prompt}\n\nThe following is an abstract.\n\n{question}",
    "lcc": "{compressed_prompt}",
    "repobench-p": "{compressed_prompt}{question}"
}

def get_prompt(ctxs, qas, dataset, use_org_idx=True):
    if 'longbench' in dataset:
        dataname = '_'.join(dataset.split('_')[1:])
        
        prompt = ""
        cur_ctx_id = 0
        for ctx in ctxs:
            if int(ctx['id'].split('-')[0]) == cur_ctx_id:
                if prompt == "":
                    prompt = ctx['text']
                else:
                    prompt = prompt + " " + ctx['text']
            else:
                prompt = prompt + '\n' + ctx['text']
                cur_ctx_id = int(ctx['id'].split('-')[0])
        # prompt = _dict[dataname].format(compressed_prompt=prompt, question=qas['question'])
        prompt = _dict_wo_instruction[dataname].format(compressed_prompt=prompt, question=qas['question'])
        # prompt = prompt.strip('\n') + "\n\n" + qas['question']
        # prompt = "\n".join([ctx['text'] for ctx in ctxs] + [qas['question']])
    else:
        ctxs_formatted = []
        for c_i, ctx in enumerate(ctxs):
            idx = ctx['org_idx'] if use_org_idx else c_i + 1
            ctxs_formatted.append(f"Document [{idx}](Title: {ctx['title']}) {ctx['text']}")
        prompt = INSTRUCTION + '\n\n' + '\n'.join(ctxs_formatted) + '\n\n' + QUESTION_TEMPLATE.format(qas['question'])

    return prompt

def main(args, logger):
    chatgpt_tok = tiktoken.encoding_for_model("gpt-3.5-turbo")
    t5_tok = T5Tokenizer.from_pretrained('t5-base')

    #################################### Loading input dataset
    if args.input_path.endswith('.jsonl'):
        test_data_org = []
        with open(args.input_path, 'r') as f:
            for line in f:
                test_data_org.append(json.loads(line))
    elif args.input_path.endswith('.jsonl.gz'):
        test_data_org = []
        with xopen(args.input_path, 'r') as f:
            for line in f:
                test_data_org.append(json.loads(line))
    elif args.input_path.endswith('.json'):
        test_data_org = json.load(open(args.input_path))
    else:
        raise ValueError("Please provide a valid input file path")

    # Set original index for each context
    for qas in test_data_org:
        for c_i, ctx in enumerate(qas['ctxs']):
            ctx['org_idx'] =  c_i + 1

    test_data = deepcopy(test_data_org)
    ##########################################################


    ### Loading token scores or FiD model
    if args.use_token_scores:
        with open(args.token_scores_path, 'rb') as f:
            token_scores_list = pickle.load(f)
    else:
        ## Define FiD model to predict token_scores
        raise NotImplementedError("Please provide token_scores_list or implement FiD model to predict token_scores")

    if 'longbench' in args.dataset:
        args.n_contexts = None
        pattern_str = 'context:'
        logger.info(f"> As the dataset is longbench, n_contexts is set to None and pattern_str is set to {pattern_str}")
    else:
        pattern_str = 'title:'
        logger.info(f"> n_contexts: {args.n_contexts}, pattern_str: {pattern_str}")

    prompt_len_list_org = []
    for qas in test_data_org:
        prompt = get_prompt(qas['ctxs'][:args.n_contexts], qas, args.dataset, args.use_org_idx)
        prompt_len_list_org.append(len(chatgpt_tok.encode(prompt)))
    org_avg_len = np.mean(prompt_len_list_org)

    # if not args.comp_ctx: args.ctx_score_cumsum = 1.0
    # if not args.comp_sent: args.sent_low, args.sent_high = 1.0, 1.0
    if not args.comp_tok: args.tok_lamb = 1.0

    # if args.ctx_score_cumsum == 1.0 and not args.use_gini: args.comp_ctx = False
    # if args.sent_low == 1.0 and args.sent_high == 1.0 and not args.use_gini: args.comp_sent = False
    if args.tok_lamb == 1.0: args.comp_tok = False
    if args.sent_comp_ratio == 0: args.comp_sent = False

    logger.info("###############################################")
    logger.info(f"> Dataset: {args.dataset}")
    logger.info(f"> Start compressing {args.input_path}")
    logger.info(f"> Original avg_len: {org_avg_len:.2f}")
    logger.info(f"> Target comprrssion settings, target_length: {args.target_length}")
    logger.info(f"> comp_ctx: {args.comp_ctx}, comp_sent: {args.comp_sent} ({args.sent_comp_ratio}), comp_tok: {args.comp_tok}")
    # logger.info(f"> ctx_score_cumsum: {args.ctx_score_cumsum}, sent_low: {args.sent_low}, sent_high: {args.sent_high}, tok_lamb: {args.tok_lamb}")
    ### Start compressing
    all_len_change_tracker = []
    start_time = time()
    for qas_i, qas in enumerate(tqdm(test_data, dynamic_ncols=True)):
        len_change_tracker = []
        ctxs = qas['ctxs'][:args.n_contexts]
        qas['ctxs'] = ctxs
        org_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
        len_org_prompt = len(chatgpt_tok.encode(org_prompt))
        compression_ratio = args.target_length / len_org_prompt

        if compression_ratio > 1.0:
            print(f"Skipping {'_'.join(args.dataset.split('_')[1:])}-{qas_i} as target_length is larger than original prompt length")
            qas['compressed_prompt'] = org_prompt
        else:
            ## Emprically, we found that removing a few tokens do not hurt the effectivness. 
            ## eg., sent_comp_ratio = 0.15 --> sent_remove_tokens/ctx_remove_tokens = 0.15
            ## Tokens need to be removed = (1 - compression_ratio) * len_org_prompt
            
            if args.use_token_scores:
                batch_scores, batch_token_ids = token_scores_list[qas_i]
                batch_scores = np.array(batch_scores) ## result shape: (20, max_len)
                batch_token_ids = batch_token_ids.squeeze(0).numpy() ## (batch_size (1), n_passage, max_len) --> (n_passage, max_len)
            else:
                pass
            
            ### If using gini index, set args.ctx_score_cumsum & args.sent_low
            # from IPython import embed; embed()
            if args.use_gini:
                norm_ctx_scores = get_ctx_scores(batch_scores, batch_token_ids, args.ctx_score_mode, args.question_mode, args.include_end_token, t5_tok, pattern_str)
                ctx_gini = gini(norm_ctx_scores)
                if ctx_gini > 0.4: ## High inequality --> high compression ratio for contexts
                    compression_ratio = 0.15
                else:
                    compression_ratio = 0.3 ## 0.4
            else:
                sent_comp_ratio = args.sent_comp_ratio

            total_remove_tokens = (1 - compression_ratio) * len_org_prompt
            ctx_comp_len = len_org_prompt - int(total_remove_tokens / (sent_comp_ratio + 1))
            sent_comp_len = int(ctx_comp_len - args.target_length)

            ctx_indices_sorted = None
            if args.comp_ctx:
                # args.target_length
                batch_scores, batch_token_ids, ctxs, ctx_indices_sorted = compress_contexts(args,
                                                                                            batch_scores,
                                                                                            batch_token_ids,
                                                                                            t5_tok,
                                                                                            ctxs,
                                                                                            ctx_comp_len,
                                                                                            pattern_str=pattern_str)
                compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                len_after_ctx_comp = len(chatgpt_tok.encode(compressed_prompt))
                len_change_tracker.append(len_after_ctx_comp)

            if args.comp_sent:
                sent_comp_len = len_after_ctx_comp - args.target_length
                batch_scores, batch_token_ids, ctxs = compress_sentences(batch_scores,
                                                                         batch_token_ids,
                                                                         t5_tok,
                                                                         ctxs,
                                                                         ctx_indices_sorted,
                                                                         sent_comp_len,
                                                                         args.adaptive_sent_comp,
                                                                         pattern_str)

                
                compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                len_after_sent_comp = len(chatgpt_tok.encode(compressed_prompt))
                len_change_tracker.append(len_after_sent_comp)

            # print(f"{len_org_prompt} {len_after_ctx_comp} {len_after_sent_comp}")
            if args.comp_tok:
                ctxs = compress_tokens(batch_scores,
                                       batch_token_ids,
                                       ctxs,
                                       tok_comp_rate,
                                       args.adaptive_tok_comp,
                                       t5_tok)
                    
                compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

            all_len_change_tracker.append(len_change_tracker)

            compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
            qas['compressed_prompt'] = compressed_prompt

    logger.info(f"Done compressing. time taken: {time() - start_time:.2f}s")

    ### logging len_change_tracker
    all_len_change_tracker = np.array(all_len_change_tracker)
    tracking_log_text = f"> Original avg_len: {org_avg_len:.2f} "
    index, fields = 0, []
    if args.comp_ctx:
        fields.append(f"comp_ctx: {all_len_change_tracker[:, index].mean():.2f}")
        index += 1
    if args.comp_sent:
        fields.append(f"comp_sent({args.sent_comp_ratio}): {all_len_change_tracker[:, index].mean():.2f}")
        index += 1
    if args.comp_tok:
        fields.append(f"comp_tok({args.tok_lamb}): {all_len_change_tracker[:, index].mean():.2f}")

    if not fields:
        raise ValueError("At least one compression method should be enabled")
    tracking_log_text += " ".join(fields)
    logger.info(tracking_log_text)

    ### Save compressed data
    # if args.use_gini:
    #     output_file_name = f"fid_gini{args.ctx_gini_standard}_ctx{args.ctx_score_cumsum_gini_low}-{args.ctx_score_cumsum_gini_high}_sent{args.sent_low_gini_high}-{args.sent_low_gini_low}"
    # else:
    #     output_file_name = f"fid_ctx{args.comp_ctx}{args.ctx_score_cumsum}_sent{args.comp_sent}{args.sent_low}-{args.sent_high}_tok{args.comp_tok}{args.tok_lamb}"
    if args.use_gini:
        output_file_name = f"fid_gini{args.ctx_gini_standard}_ctx{args.ctx_score_cumsum_gini_low}-{args.ctx_score_cumsum_gini_high}_sent{args.sent_low_gini_high}-{args.sent_low_gini_low}"
    else:
        output_file_name = f"fid_{args.target_length}_ctx{args.comp_ctx}_sent{args.comp_sent}{args.sent_comp_ratio}_tok{args.comp_tok}{args.tok_lamb}_orgidx{args.use_org_idx}"

    if 'longbench' in args.dataset: ## args.dataset == 'longbench_samsum' or 'dpr_nq_20' / args.output_root == 'compressed_qa_data/0423'
        output_path = os.path.join(f"{args.output_root}", f"{args.dataset}_{output_file_name}.json")
    else:            
        output_path = os.path.join(args.output_root, f"{args.dataset}_{args.n_contexts}", f"{output_file_name}.jsonl.gz")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"> Saving compressed data to {output_path}")
    with xopen(output_path, 'wt') as outf:
        if output_path.endswith('.jsonl.gz'):
                for qas in test_data:
                    outf.write(json.dumps(qas) + '\n')
        elif output_path.endswith('.json'):
            json.dump(test_data, outf, indent=4)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="qa_data", required=True,
                        help="A path of json or jsonl file containing a list of qas") ## e.g., "qa_data/test_20.json"
    parser.add_argument('--output_root', type=str, default="compressed_qa_data")
    parser.add_argument('--dataset', type=str, default="dpr_nq_20") ## e.g., dpr_20, 20_gold_at_0
    parser.add_argument('--n_contexts', type=int, default=20)
    parser.add_argument('--use_token_scores', action='store_true', default=False,
                        help="Whether to use token_scores_list predicted in advance")
    parser.add_argument('--token_scores_path', type=str, default="qa_data/test_sent_20_192_nltk.json", required=False,
                        help="A path of token scores containing a list of tuples of ((input_ids, scores))") ## e.g., "token_scores/token_scores_list_dpr_20_oneContextFalse.pkl"
    parser.add_argument('--use_gini', action='store_true', default=False,
                        help="Whether to use gini index of ctx scores")
    parser.add_argument('--ctx_gini_standard', type=float, default=0.4, required=False,
                        help="Gini index standard to adjust compression ratios")
    
    parser.add_argument('--ctx_score_cumsum_gini_low', type=float, default=0.3, required=False)
    parser.add_argument('--ctx_score_cumsum_gini_high', type=float, default=0.4, required=False)
    parser.add_argument('--sent_low_gini_low', type=float, default=0.3, required=False)
    parser.add_argument('--sent_low_gini_high', type=float, default=0.8, required=False)
    
    parser.add_argument('--target_length', type=int, default=False,
                        help="Number of target tokens for the compressed prompt")

    parser.add_argument('--comp_ctx', action='store_true', default=False,
                        help="Whether to use context compression")
    parser.add_argument('--do_sort_ctx', action='store_true', default=False,
                        help="Whether to apply sorting in context compression")
    parser.add_argument('--use_org_idx', action='store_true', default=False,
                        help="Whether to use original index for each context")
    parser.add_argument('--ctx_score_mode', type=str, default='mean', required=False,
                        help="Mode to calculate ctx score")
    parser.add_argument('--include_end_token', action='store_true', default=False,
                        help="Whether to include end token in ctx score calculation")
    parser.add_argument('--question_mode', type=str, default='exclude', choices=['exclude', 'include', 'only'],
                        help="Mode to include question in ctx score calculation")

    parser.add_argument('--comp_sent', action='store_true', default=False,
                        help="Whether to use sentence compression")
    parser.add_argument('--sent_comp_ratio', type=float, default=0.15, required=False,
                        help="Ratio of sentence compression")
    parser.add_argument('--adaptive_sent_comp', type=bool, default=False, required=False,
                        help="Whether to use adaptive sentence compression")

    parser.add_argument('--comp_tok', action='store_true', default=False,
                        help="Whether to use token compression")
    parser.add_argument('--adaptive_tok_comp', type=bool, default=False, required=False,
                        help="Whether to use adaptive token compression")
    parser.add_argument('--tok_lamb', type=float, default=0.95, required=False,
                        help="Cumulative normalized sentence score to keep for the highest context")

    args = parser.parse_args()
    
    ## Write logger.info into log.txt
    logger = logging.getLogger(__name__)
    # handler = logging.FileHandler("log.txt")
    handlers = [logging.FileHandler("log.txt"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    main(args, logger)
    logger.info("###################################################")