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
from src_comp.compress_utils import *
from transformers import T5TokenizerFast, AutoTokenizer


def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities))

def main(args, logger):
    chatgpt_tok = tiktoken.encoding_for_model("gpt-3.5-turbo")
    scorer_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    # scorer_tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', use_fast=False)

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
    elif 'nq' in args.dataset:
        pattern_str = 'title:'
        logger.info(f"> Compressing Natural Questions, n_contexts: {args.n_contexts}, pattern_str: {pattern_str}")
    else:
        pattern_str = 'context:'
        logger.info(f"> n_contexts: {args.n_contexts}, pattern_str: {pattern_str}")

    prompt_len_list_org = []
    for qas in test_data_org:
        prompt = get_prompt(qas['ctxs'][:args.n_contexts], qas, args.dataset, args.use_org_idx)
        prompt_len_list_org.append(len(chatgpt_tok.encode(prompt)))
    org_avg_len = np.mean(prompt_len_list_org)
    logger.info(f"> Original avg_len: {args.dataset} {org_avg_len:.2f}")

    if not args.comp_tok or args.e_tok == 0: args.comp_tok = False
    if args.sent_comp_ratio == 1: args.comp_ctx = False
    if not args.comp_sent or args.sent_comp_ratio == 0:
        args.comp_sent = False
        args.sent_comp_ratio = 0

    logger.info("###############################################")
    logger.info(f"> Dataset: {args.dataset}")
    logger.info(f"> Start compressing {args.input_path}")
    logger.info(f"> Token score path: {args.token_scores_path}")
    logger.info(f"> Original avg_len: {org_avg_len:.2f}")
    logger.info(f"> Target comprrssion settings, target_length: {args.target_length}")

    if args.use_gini:
        args.sent_comp_ratio = f"gini_{args.ctx_gini_standard}_{args.raw_for_high_gini}_{args.raw_for_low_gini}"
    logger.info(f"> comp_ctx: {args.comp_ctx} ({1 - args.sent_comp_ratio}), comp_sent: {args.comp_sent} ({args.sent_comp_ratio}), comp_tok: {args.comp_tok}")
    ### Start compressing
    all_len_change_tracker = []
    start_time = time()
    log = open('log_compress_time_measure.txt', 'a')
    s = 0
    cnt = 0
    for qas_i, qas in enumerate(tqdm(test_data, dynamic_ncols=True)):
        len_change_tracker = []
        ctxs = qas['ctxs'][:args.n_contexts]
        question = qas['question']
        if qas.get('all_classes') is not None:
            force_tokens = qas['all_classes']
        else:
            force_tokens = None
        
        qas['ctxs'] = ctxs
        org_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
        len_org_prompt = len(chatgpt_tok.encode(org_prompt))
        if 'longbench' in args.dataset:
            dummy_ctxs = []
        elif 'nq' in args.dataset:
            num_ctxs_estimate = int(len(ctxs) * (args.target_length / len_org_prompt))
            dummy_ctxs = [{'title': ctx['title'], 'text': "", 'org_idx': ctx['org_idx']} for ctx in ctxs[:num_ctxs_estimate + 1]]
        else:
            dummy_ctxs = []
        len_wo_ctxs = len(GPT_TOKENIZER.encode(get_prompt(dummy_ctxs, {'question': question}, args.dataset, use_org_idx=args.use_org_idx)))
        # coarse_target_length = (args.target_length - len_wo_ctxs) * 1/args.tok_lamb + len_wo_ctxs
        coarse_target_length = args.target_length + args.e_tok
        total_coarse_remove_tokens = len_org_prompt - coarse_target_length

        if qas_i == 0:
            logger.info(f"> coarse_target_length: {coarse_target_length:.2f}, total_coarse_remove_tokens: {total_coarse_remove_tokens:.2f}")

        len_after_ctx_comp = len_org_prompt
        len_after_sent_comp = len_org_prompt

        if total_coarse_remove_tokens < 0:
            compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
            # print(f"Skipping {'_'.join(args.dataset.split('_')[1:])}-{qas_i} as target_length is larger than original prompt length")
            if args.comp_ctx:
                len_change_tracker.append(len_org_prompt)
            if args.comp_sent:
                len_change_tracker.append(len_org_prompt)

            if args.comp_tok:
                if args.use_token_scores:
                    batch_scores, batch_token_ids = token_scores_list[qas_i]
                    batch_scores = np.array(batch_scores) ## result shape: (20, max_len)
                    batch_token_ids = batch_token_ids.squeeze(0).numpy() ## (batch_size (1), n_passage, max_len) --> (n_passage, max_len)
                else:
                    pass
                cur_len = len_after_sent_comp if args.comp_sent else len_after_ctx_comp ## cur_len = INSTRUCTION + QUESTION + CONTEXTS / tok_lamb only consider context

                if cur_len - len_wo_ctxs > 0:
                    tok_lamb = (args.target_length - len_wo_ctxs) / (cur_len - len_wo_ctxs)
                else:
                    tok_lamb = 1.0

                if not ctxs or tok_lamb >= 1.0:
                    len_change_tracker.append(len_org_prompt)
                    pass
                else:
                    titles = [ctx['title'] for ctx in ctxs]
                    ctx_start_indices = get_ctx_start_indices(scorer_tokenizer, question, titles, pattern_str)
                    batch_eos_token_idx, batch_len_context = [], []
                    for ctx_i, token_ids in enumerate(batch_token_ids):
                        eos_token_idx = np.where(token_ids == 1)[0][-1]
                        batch_eos_token_idx.append(eos_token_idx)
                        batch_len_context.append(eos_token_idx - ctx_start_indices[ctx_i])

                    batch_scores = [batch_scores[i][ctx_start_indices[i]:eos_token_idx] for i, eos_token_idx in enumerate(batch_eos_token_idx)]
                    batch_token_ids = [batch_token_ids[i][ctx_start_indices[i]:eos_token_idx] for i, eos_token_idx in enumerate(batch_eos_token_idx)]
                    # tok_target_tokens_len = np.concatenate(batch_scores).shape[0] * (1 - tok_lamb)
                    ctxs = compress_tokens(args,
                                           batch_scores,
                                           batch_token_ids,
                                           ctxs,
                                           tok_lamb,
                                           args.adaptive_tok_comp,
                                           scorer_tokenizer,
                                           question,
                                           force_tokens,
                                           true_target=args.target_length-len_wo_ctxs)
                    compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                    len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))
                

        else:
            ## Emprically, we found that removing a few tokens do not hurt the effectivness. 
            ## eg., sent_comp_ratio = 0.15 --> sent_remove_tokens/ctx_remove_tokens = 0.15
            
            if args.use_token_scores:
                batch_scores, batch_token_ids = token_scores_list[qas_i]
                batch_scores = np.array(batch_scores) ## result shape: (20, max_len)
                batch_token_ids = batch_token_ids.squeeze(0).numpy() ## (batch_size (1), n_passage, max_len) --> (n_passage, max_len)
            else:
                pass
            
            ctx_indices_sorted = list(range(len(ctxs)))
            if args.comp_ctx:
                ctx_comp_len = total_coarse_remove_tokens * (1 - args.sent_comp_ratio) # |E_C|
                ctx_target_len = int(len_org_prompt - ctx_comp_len) ## |P_C| - |E_C|

                # args.target_length
                batch_scores, batch_token_ids, ctxs, ctx_indices_sorted = compress_contexts(args,
                                                                                            batch_scores,
                                                                                            batch_token_ids,
                                                                                            scorer_tokenizer,
                                                                                            ctxs,
                                                                                            ctx_target_len,
                                                                                            question,
                                                                                            pattern_str)
                compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                len_after_ctx_comp = len(chatgpt_tok.encode(compressed_prompt))
                len_change_tracker.append(len_after_ctx_comp)

            elif args.do_sort_ctx:
                titles = [ctx['title'] for ctx in ctxs]
                ctx_scores = get_ctx_scores(batch_scores, args.ctx_score_mode, args.question_mode, args.include_end_token, scorer_tokenizer, question, titles, pattern_str)
                ctx_indices_sorted = np.argsort(ctx_scores)[::-1].tolist()
                ctxs = [ctxs[i] for i in ctx_indices_sorted]
                batch_scores = [batch_scores[i] for i in ctx_indices_sorted]
                batch_token_ids = [batch_token_ids[i] for i in ctx_indices_sorted]

            if args.comp_sent:
                sent_comp_len = len_after_ctx_comp - coarse_target_length ## E_sent
                if sent_comp_len > 0:
                    batch_scores, batch_token_ids, ctxs = compress_sentences(args,
                                                                             batch_scores,
                                                                             batch_token_ids,
                                                                             scorer_tokenizer,
                                                                             ctxs,
                                                                             ctx_indices_sorted,
                                                                             sent_comp_len,
                                                                             args.adaptive_sent_comp,
                                                                             question,
                                                                             "context:",
                                                                             args.pow,
                                                                             args.constraint_1_sent)
                
                compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                len_after_sent_comp = len(chatgpt_tok.encode(compressed_prompt))
                len_change_tracker.append(len_after_sent_comp)

            if args.comp_tok:
                cur_len = len_after_sent_comp if args.comp_sent else len_after_ctx_comp ## cur_len = INSTRUCTION + QUESTION + CONTEXTS / tok_lamb only consider context
                if 'longbench' in args.dataset:
                    dummy_ctxs = []
                else:
                    dummy_ctxs = [{'title': ctx['title'], 'text': "", 'org_idx': ctx['org_idx']} for ctx in ctxs]
                len_wo_ctxs = len(GPT_TOKENIZER.encode(get_prompt(dummy_ctxs, {'question': question}, args.dataset, use_org_idx=args.use_org_idx)))
                
                if cur_len - len_wo_ctxs > 0:
                    tok_lamb = (args.target_length - len_wo_ctxs) / (cur_len - len_wo_ctxs)
                    # tok_lamb = (args.target_length - len_wo_ctxs) / np.concatenate(batch_scores).shape[0]

                else:
                    tok_lamb = 1.0

                if not ctxs or tok_lamb >= 1.0:
                    len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))
                    pass
                else:
                    if not args.comp_sent:
                        titles = [ctx['title'] for ctx in ctxs]
                        ctx_start_indices = get_ctx_start_indices(scorer_tokenizer, question, titles, pattern_str)
                        batch_eos_token_idx, batch_len_context = [], []
                        for ctx_i, token_ids in enumerate(batch_token_ids):
                            eos_token_idx = np.where(token_ids == 1)[0][-1]
                            batch_eos_token_idx.append(eos_token_idx)
                            batch_len_context.append(eos_token_idx - ctx_start_indices[ctx_i])

                        batch_scores = [batch_scores[i][ctx_start_indices[i]:eos_token_idx] for i, eos_token_idx in enumerate(batch_eos_token_idx)]
                        batch_token_ids = [batch_token_ids[i][ctx_start_indices[i]:eos_token_idx] for i, eos_token_idx in enumerate(batch_eos_token_idx)]
                    # tok_target_tokens_len = np.concatenate(batch_scores).shape[0] * (1 - tok_lamb)
                    ctxs = compress_tokens(args,
                                           batch_scores,
                                           batch_token_ids,
                                           ctxs,
                                           tok_lamb,
                                           args.adaptive_tok_comp,
                                           scorer_tokenizer,
                                           question,
                                           force_tokens,
                                           true_target=args.target_length-len_wo_ctxs)
                    compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
                    len_change_tracker.append(len(chatgpt_tok.encode(compressed_prompt)))

        all_len_change_tracker.append(len_change_tracker)
        compressed_prompt = get_prompt(ctxs, qas, args.dataset, args.use_org_idx)
        qas['compressed_prompt'] = compressed_prompt

    logger.info(f"Done compressing. time taken: {time() - start_time:.2f}s")
    log.write(f"{args.dataset} {time() - start_time:.2f}s\n")
    log.close()
    ### logging len_change_tracker
    all_len_change_tracker = np.array(all_len_change_tracker)
    tracking_log_text = f"> Original avg_len: {org_avg_len:.2f} "
    index, fields = 0, []
    if args.comp_ctx:
        fields.append(f"comp_ctx({1 - args.sent_comp_ratio}): {all_len_change_tracker[:, index].mean():.2f}")
        index += 1
    if args.comp_sent:
        fields.append(f"comp_sent({args.sent_comp_ratio}): {all_len_change_tracker[:, index].mean():.2f}")
        index += 1
    if args.comp_tok:
        fields.append(f"comp_tok({args.e_tok}): {all_len_change_tracker[:, index].mean():.2f}")

    fields.append(f"(target_length: {args.target_length})")
    if not fields:
        raise ValueError("At least one compression method should be enabled")
    tracking_log_text += " ".join(fields)
    logger.info(tracking_log_text)


    output_file_name = f"fid_{args.target_length}_ctx{args.comp_ctx}_sort{args.do_sort_ctx}_sent{args.comp_sent}{args.sent_comp_ratio}_{args.pow}_tok{args.comp_tok}{args.e_tok}"

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
    parser.add_argument('--ctx_gini_standard', type=float, default=0.3, required=False,
                        help="Gini index standard to adjust compression ratios")
    parser.add_argument('--raw_for_low_gini', type=float, default=0.2, required=False)
    parser.add_argument('--raw_for_high_gini', type=float, default=0.15, required=False)
    parser.add_argument('--pow', type=int, default=2) ## e.g., dpr_20, 20_gold_at_0


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
    parser.add_argument('--question_mode', type=str, default='include', choices=['exclude', 'include', 'only'],
                        help="Mode to include question in ctx score calculation")

    parser.add_argument('--comp_sent', action='store_true', default=False,
                        help="Whether to use sentence compression")
    parser.add_argument('--sent_comp_ratio', type=float, default=0.15, required=False,
                        help="Ratio of sentence compression")
    parser.add_argument('--adaptive_sent_comp', action='store_true', default=False, required=False,
                        help="Whether to use adaptive sentence compression")
    parser.add_argument('--constraint_1_sent', type=bool, default=False, required=False,
                        help="Whether to use adaptive sentence compression")

    parser.add_argument('--comp_tok', action='store_true', default=False,
                        help="Whether to use token compression")
    parser.add_argument('--e_tok', type=int, default=0, required=False,
                        help="Number of tokens to be eliminated")
    parser.add_argument('--adaptive_tok_comp', type=bool, default=False, required=False,
                        help="Whether to use adaptive token compression")

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