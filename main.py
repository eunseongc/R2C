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


INSTRUCTION = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
QUESTION_TEMPLATE = "Question: {}\nAnswer:"

def get_prompt(ctxs, qas):
    ctxs_formatted = [f"Document [{ctx['org_idx']}](Title: {ctx['title']}) {ctx['text']}" for ctx in qas['ctxs']]
    prompt = INSTRUCTION + '\n\n' + '\n'.join(ctxs_formatted) + '\n\n' + QUESTION_TEMPLATE.format(qas['question'])
    return prompt

def main(args, logger):
    chatgpt_tok = tiktoken.encoding_for_model("gpt-3.5-turbo")

    if args.input_path.endswith('.jsonl'):
        test_data_org = []
        with open(args.input_path, 'r') as f:
            for line in f:
                test_data_org.append(json.loads(line))
    elif args.input_path.endswith('.json'):
        test_data_org = json.load(open(args.input_path))
        for qas in test_data_org:
            for c_i, ctx in enumerate(qas['ctxs']):
                ctx['isgold'] = False
                ctx['org_idx'] =  c_i + 1

    if args.use_token_scores:
        with open(args.token_scores_path, 'rb') as f:
            token_scores_list = pickle.load(f)

    prompt_len_list_org = []
    for qas in test_data_org:
        ctxs = [f"Document [{ctx['org_idx']}](Title: {ctx['title']}) {ctx['text']}" for ctx in qas['ctxs']]
        prompt = INSTRUCTION + '\n\n' + '\n'.join(ctxs) + '\n\n' + QUESTION_TEMPLATE.format(qas['question'])
        prompt_len_list_org.append(len(chatgpt_tok.encode(prompt)))

    org_avg_len = np.mean(prompt_len_list_org)
    logger.info(f"Original avg_len: {org_avg_len:.2f}")

    test_data = deepcopy(test_data_org)
    t5_tok = T5Tokenizer.from_pretrained('t5-base')

    ctx_score_cumsum = 0.4
    sent_low = 0.3
    sent_high = 1.0
    token_lamb = 0.95

    all_len_change_tracker = []
    start_time = time()
    for qas_i, qas in enumerate(tqdm(test_data, dynamic_ncols=True)):
        len_change_tracker = []
        ctxs = qas['ctxs'][:args.n_contexts]

        if args.use_token_scores:
            batch_scores, batch_token_ids = token_scores_list[qas_i]
            batch_scores = np.array(batch_scores) ## result shape: (20, max_len)
            batch_token_ids = batch_token_ids.squeeze().numpy() ## result shape: (20, max_len)

        if args.comp_ctx:
            batch_scores, batch_token_ids, doc_indices_selected = compress_contexts(batch_scores,
                                                                                    batch_token_ids,
                                                                                    ctx_score_cumsum,
                                                                                    do_sort=args.do_sort_ctx)
            ctxs = [qas['ctxs'][i] for i in doc_indices_selected]
            compressed_prompt = get_prompt(ctxs, qas)
            len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

        if args.comp_sent:
            batch_scores, batch_token_ids, ctxs = compress_sentences(batch_scores,
                                             batch_token_ids,
                                             ctxs,
                                             sent_low,
                                             sent_high,
                                             t5_tok)

            compressed_prompt = get_prompt(ctxs, qas)
            len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

        if args.comp_tok:
            ctxs = compress_tokens(batch_scores,
                                         batch_token_ids,
                                         ctxs,
                                         token_lamb,
                                         t5_tok)
            
            compressed_prompt = get_prompt(ctxs, qas)
            len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

        all_len_change_tracker.append(len_change_tracker)
        
        compressed_prompt = get_prompt(ctxs, qas)
        qas['compressed_prompt'] = compressed_prompt

    logger.info(f"Done compressing. time taken: {time() - start_time:.2f}s")

    # logging len_change_tracker
    all_len_change_tracker = np.array(all_len_change_tracker)
    ## org_avg_len: {org_avg_len} --> comp_avg_len: {comp_avg_len}
    comp_avg_len = all_len_change_tracker[:, -1].mean()
    logger.info(f"Original avg_len: {org_avg_len:.2f} --> Compressed avg_len: {comp_avg_len:.2f}")



    output_file_name = f"fid_doc{args.ctx_score_cumsum}_sl{sent_low}_sh{sent_high}_tl{token_lamb}.jsonl.gz"
    output_path= os.path.join(args.output_root, f"nq_{args.n_contexts}", output_file_name)
    if not os.path.exists(os.path.dirname(output_path)):
        ## Consider if output_root also not exist.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
    parser.add_argument('--token_lamb', type=float, default=0.95, required=False,
                        help="Cumulative normalized sentence score to keep for the highest context")

    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    main(args, logger)
    