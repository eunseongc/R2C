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

def get_prompt(ctxs, qas):
    ctxs_formatted = [f"Document [{ctx['org_idx']}](Title: {ctx['title']}) {ctx['text']}" for ctx in ctxs]
    prompt = INSTRUCTION + '\n\n' + '\n'.join(ctxs_formatted) + '\n\n' + QUESTION_TEMPLATE.format(qas['question'])
    return prompt

def main(args, logger):
    chatgpt_tok = tiktoken.encoding_for_model("gpt-3.5-turbo")

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

    for qas in test_data_org:
        for c_i, ctx in enumerate(qas['ctxs']):
            ctx['org_idx'] =  c_i + 1

    if args.use_token_scores:
        with open(args.token_scores_path, 'rb') as f:
            token_scores_list = pickle.load(f)
    else:
        ## Define FiD model to predict token_scores
        raise NotImplementedError("Please provide token_scores_list or implement FiD model to predict token_scores")

    prompt_len_list_org = []
    for qas in test_data_org:
        prompt = get_prompt(qas['ctxs'], qas)
        prompt_len_list_org.append(len(chatgpt_tok.encode(prompt)))

    org_avg_len = np.mean(prompt_len_list_org)

    test_data = deepcopy(test_data_org)
    t5_tok = T5Tokenizer.from_pretrained('t5-base')

    if not args.comp_ctx: args.ctx_score_cumsum = 1.0
    if not args.comp_sent: args.sent_low, args.sent_high = 1.0, 1.0
    if not args.comp_tok: args.tok_lamb = 1.0

    if args.ctx_score_cumsum == 1.0 and not args.use_gini: args.comp_ctx = False
    if args.sent_low == 1.0 and args.sent_high == 1.0 and not args.use_gini: args.comp_sent = False
    if args.tok_lamb == 1.0: args.comp_tok = False

    logger.info("###################################################")
    logger.info(f"> Start compressing {args.input_path}")
    logger.info(f"> Original avg_len: {org_avg_len:.2f}")
    logger.info(f"> comp_ctx: {args.comp_ctx}, comp_sent: {args.comp_sent}, comp_tok: {args.comp_tok}")
    logger.info(f"> ctx_score_cumsum: {args.ctx_score_cumsum}, sent_low: {args.sent_low}, sent_high: {args.sent_high}, tok_lamb: {args.tok_lamb}")

    ## Start compressing
    all_len_change_tracker = []
    start_time = time()
    for qas_i, qas in enumerate(tqdm(test_data, dynamic_ncols=True)):
        len_change_tracker = []
        ctxs = qas['ctxs'][:args.n_contexts]

        if args.use_token_scores:
            batch_scores, batch_token_ids = token_scores_list[qas_i]
            batch_scores = np.array(batch_scores) ## result shape: (20, max_len)
            batch_token_ids = batch_token_ids.squeeze().numpy() ## result shape: (20, max_len)
        else:
            pass
    
        ## If using gini index, set args.ctx_score_cumsum & args.sent_low
        if args.use_gini:
            norm_ctx_scores = get_ctx_scores(batch_scores)
            ctx_gini = gini(norm_ctx_scores)
            if ctx_gini > args.ctx_gini_standard: ## High inequality --> high compression ratio for contexts
                args.ctx_score_cumsum = args.ctx_score_cumsum_gini_low ## 0.3
                args.sent_low = args.sent_low_gini_high ## 0.8
            else:
                args.ctx_score_cumsum = args.ctx_score_cumsum_gini_high ## 0.4
                args.sent_low = args.sent_low_gini_low ## 0.3

        if args.comp_ctx:
            batch_scores, batch_token_ids, ctx_indices_selected = compress_contexts(batch_scores,
                                                                                    batch_token_ids,
                                                                                    args.ctx_score_cumsum,
                                                                                    do_sort_ctx=args.do_sort_ctx)
            ctxs = [qas['ctxs'][i] for i in ctx_indices_selected]
            compressed_prompt = get_prompt(ctxs, qas)
            len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

        if args.comp_sent:
            batch_scores, batch_token_ids, ctxs = compress_sentences(batch_scores,
                                                                     batch_token_ids,
                                                                     ctxs,
                                                                     args.sent_low,
                                                                     args.sent_high,
                                                                     t5_tok)

            compressed_prompt = get_prompt(ctxs, qas)
            len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

        if args.comp_tok:
            ctxs = compress_tokens(batch_scores,
                                   batch_token_ids,
                                   ctxs,
                                   args.tok_lamb,
                                   t5_tok)
            
            compressed_prompt = get_prompt(ctxs, qas)
            len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))
        
        all_len_change_tracker.append(len_change_tracker)
        compressed_prompt = get_prompt(ctxs, qas)
        qas['compressed_prompt'] = compressed_prompt

    logger.info(f"Done compressing. time taken: {time() - start_time:.2f}s")

    ### logging len_change_tracker
    all_len_change_tracker = np.array(all_len_change_tracker)
    tracking_log_text = f"> Original avg_len: {org_avg_len:.2f} "
    index, fields = 0, []
    if args.comp_ctx:
        fields.append(f"comp_ctx ({args.ctx_score_cumsum}): {all_len_change_tracker[:, index].mean():.2f}")
        index += 1
    if args.comp_sent:
        fields.append(f"comp_sent({args.sent_low}-{args.sent_high}): {all_len_change_tracker[:, index].mean():.2f}")
        index += 1
    if args.comp_tok:
        fields.append(f"comp_tok: {all_len_change_tracker[:, index].mean():.2f}")

    if not fields:
        raise ValueError("At least one compression method should be enabled")
    tracking_log_text += " ".join(fields)
    logger.info(tracking_log_text)

    ### Save compressed data
    if args.use_gini:
        output_file_name = f"fid_gini{args.ctx_gini_standard}_ctx{args.ctx_score_cumsum_gini_low}-{args.ctx_score_cumsum_gini_high}_sent{args.sent_low_gini_high}-{args.sent_low_gini_low}.jsonl.gz"
    else:
        output_file_name = f"fid_ctx{args.comp_ctx}{args.ctx_score_cumsum}_sent{args.comp_sent}{args.sent_low}-{args.sent_high}_tok{args.comp_tok}{args.tok_lamb}.jsonl.gz"
    if args.output_root == "compressed_qa_data":
        output_path = os.path.join(args.output_root, f"nq_{args.n_contexts}", output_file_name)
    else:
        output_path = os.path.join(args.output_root, output_file_name)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"> Saving compressed data to {output_path}")       
    with xopen(output_path, 'wt') as f:
        for qas in test_data:
            f.write(json.dumps(qas) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="open_domain_data/nq/test_sent_20_192_nltk.json", required=True,
                        help="A path of json or jsonl file containing a list of qas") ## e.g., open_domain_data/nq/test_sent_20_192_nltk.json
    parser.add_argument('--output_root', type=str, default="compressed_qa_data")
    parser.add_argument('--n_contexts', type=int, default=20)
    parser.add_argument('--use_token_scores', action='store_true', default=False,
                        help="Whether to use token_scores_list predicted in advance")
    parser.add_argument('--token_scores_path', type=str, default="open_domain_data/nq/test_sent_20_192_nltk.json", required=False,
                        help="A path of token scores containing a list of tuples of ((input_ids, scores))") ## e.g., open_domain_data/nq/test_sent_20_192_nltk.json
    parser.add_argument('--use_gini', action='store_true', default=False,
                        help="Whether to use gini index of ctx scores")
    parser.add_argument('--ctx_gini_standard', type=float, default=0.4, required=False,
                        help="Gini index standard to adjust compression ratios")
    parser.add_argument('--ctx_score_cumsum_gini_low', type=float, default=0.3, required=False,
                        help="Gini index standard to adjust compression ratios")
    parser.add_argument('--ctx_score_cumsum_gini_high', type=float, default=0.4, required=False,
                        help="Gini index standard to adjust compression ratios")
    parser.add_argument('--sent_low_gini_low', type=float, default=0.3, required=False,
                        help="Gini index standard to adjust compression ratios")
    parser.add_argument('--sent_low_gini_high', type=float, default=0.8, required=False,
                        help="Gini index standard to adjust compression ratios")

    


    parser.add_argument('--comp_ctx', action='store_true', default=False,
                        help="Whether to use context compression")
    parser.add_argument('--do_sort_ctx', action='store_true', default=False,
                        help="Whether to apply sorting in context compression")
    parser.add_argument('--ctx_score_cumsum', type=float, default=0.4, required=False,
                        help="Cumulative sum of normalized ctx score")

    parser.add_argument('--comp_sent', action='store_true', default=False,
                        help="Whether to use sentence compression")
    parser.add_argument('--sent_low', type=float, default=0.4, required=False,
                        help="Cumulative normalized sentence score to keep for the lowest context")
    parser.add_argument('--sent_high', type=float, default=1.0, required=False,
                        help="Cumulative normalized sentence score to keep for the highest context")

    parser.add_argument('--comp_tok', action='store_true', default=False,
                        help="Whether to use token compression")
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
    