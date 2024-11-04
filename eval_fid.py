# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import torch
import argparse
import transformers
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import src_fid.slurm
import src_fid.util
import src_fid.data
import src_fid.evaluation
import src_fid.model
from src_fid.options import Options
from src_fid.ResultTable import ResultTable

def evaluate(model, dataset, dataloader, tokenizer, opt):
    simple_tokenizer = src_fid.evaluation.SimpleTokenizer()
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        assert opt.per_gpu_batch_size == 1, "Cross attention scores can only be written when batch size is 1"
        model.reset_score_storage()
        return_dict_in_generate = True
        pred_recall_dict_ca = {cut_off:[] for cut_off in opt.cut_offs}
    else:
        return_dict_in_generate = False
    total = 0
    exactmatch, recall = [], []
    span_exactmatch = []
    
    num_passages_in_decoder = []

    logger.warning(f'Eval dataset name: {opt.eval_data}')
    
    token_scores_list = []
    with torch.no_grad():
        for b_i, batch in enumerate(tqdm(dataloader)):
            (idx, labels, context_ids, context_mask, q_tokens, has_answers, task) = batch
            outputs = model.generate(input_ids=context_ids.cuda(),
                                     attention_mask=context_mask.cuda(),
                                     max_new_tokens=20,
                                     return_dict_in_generate=return_dict_in_generate,
                                     output_attentions=True,
                                     last_layer_only=opt.last_layer_only)
                        
            if return_dict_in_generate and opt.write_crossattention_scores:
                outputs, crossattention_scores, token_scores = outputs
                token_scores_list.append((token_scores, context_ids))

            answer_array = has_answers.numpy()

            if return_dict_in_generate and opt.write_crossattention_scores:
                if opt.mode == 'single':
                    ca_scores_array = crossattention_scores[:, :-1].cpu().numpy()
                else: ## opt.mode == 'pair'
                    ca_scores_array = crossattention_scores.cpu().numpy()
                ca_sorted_indices = np.argsort(ca_scores_array, axis=1)[:, ::-1]
                pred_answer_array_ca = np.take_along_axis(answer_array, ca_sorted_indices, axis=1)
            
            if return_dict_in_generate and opt.write_crossattention_scores:
                for cut_off in opt.cut_offs:
                    pred_recall_dict_ca[cut_off].extend(pred_answer_array_ca[:, :cut_off].sum(1).astype('bool').tolist())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src_fid.evaluation.ems(ans, example['answers'])
                    has_answer_score = src_fid.evaluation.has_answer(example['answers'], ans, simple_tokenizer)
                    span_exactmatch.append(has_answer_score)
                    exactmatch.append(score)
                answers = example['answers']
                has_answer = check_has_answer(answers, example['ctxs'][:opt.n_contexts], simple_tokenizer)
                recall.append(has_answer)
                q_i = b_i * opt.per_gpu_batch_size + k

                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1

            if (b_i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {b_i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average (EM) = {100 * np.mean(exactmatch):.3f}'
                    log += f' | average (Span EM) = {100 * np.mean(span_exactmatch):.3f}'
                    # log += f' | average (G-2) = {np.mean(exactmatch_G2):.3f}'

                logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | EM = {100 * np.mean(exactmatch):.2f}')
    logger.warning(f'Process rank:{opt.global_rank}, total {total} | Span EM = {100 * np.mean(span_exactmatch):.2f}')

    if 't0-base' in opt.model_path:
        opt.model_path = 't0-base'
    elif 't0-large' in opt.model_path:
        opt.model_path = 't0-large'
    elif 'flan-t5-base' in opt.model_path:
        opt.model_path = 'flan-t5-base'
    elif 'flan-t5-large' in opt.model_path:
        opt.model_path = 'flan-t5-large'
    else:
        opt.model_path = opt.model_path.split('/')[1] ## [0]: checkpoints_fid, [1]: model_name

    token_scores_path = opt.token_scores_path

    if not os.path.exists(os.path.dirname(token_scores_path)):
        os.makedirs(os.path.dirname(token_scores_path), exist_ok=True)
        
    if opt.write_crossattention_scores:
        logger.warning(f'Saving token scores to {token_scores_path}')
        with open(token_scores_path, 'wb') as f:
            pickle.dump(token_scores_list, f)

    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src_fid.util.weighted_average(np.mean(exactmatch), total, opt)
    score_span, total = src_fid.util.weighted_average(np.mean(span_exactmatch), total, opt)

    preds = {}
    if return_dict_in_generate and opt.write_crossattention_scores:
        pred_recall_dict_ca = {f'CA_Recall{k}':100 * src_fid.util.weighted_average(np.mean(v), total, opt)[0] for k, v in pred_recall_dict_ca.items()}
        preds.update(pred_recall_dict_ca)
    preds['EM'] = 100 * np.mean(score)
    preds['Span EM'] = 100 * np.mean(score_span)

    if opt.ce_mask_threshold > 0.0 or opt.reduced_n > 0:
        num_passages_in_decoder, _ = src_fid.util.weighted_average(np.mean(num_passages_in_decoder), total, opt)
        preds['Avg. #passages'] = np.round(num_passages_in_decoder, 2)

    return score, total, preds

def check_has_answer(answers, examples, simple_tokenizer):
    flag = False
    for ctx_id, ctx in enumerate(examples):
        if ctx['title'] is not None:
            text = ctx['title'] + ' ' + ctx['text']
        else:
            text = ctx['text']
        if src_fid.evaluation.has_answer(answers, text, simple_tokenizer):
            flag = True
            break
    return flag

def most_frequent(ans_list):
    return max(set(ans_list), key = ans_list.count)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the config file', default=None) ## e.g., checkpoints/nq_reader_qbos/checkpoint/best_dev
    parser.add_argument('--mode', help='data mode', default='pair') ## e.g., pair, single

    args, remaining_args = parser.parse_known_args()
    
    options = Options()
    options.add_reader_options()
    options.add_eval_options()

    if args.model_path in ['google/flan-t5-base', 'google/flan-t5-large', 'qinyuany/my-t0-base', 'qinyuany/my-t0-large']:
        opt_path = "checkpoints_fid/nq_20_0/options.json"
    else:
        opt_path = Path('/'.join(args.model_path.split('/')[:2])) / "options.json"

    if opt_path:
        with open(opt_path, 'r') as f:
            opt_dict = json.load(f)
        loaded_opt = argparse.Namespace(**opt_dict)
    else:
        print("> No options file found")
        exit(1)
    options.parser.set_defaults(**vars(loaded_opt))

    # Parse command line arguments
    # Any command line argument will overwrite the one from JSON file
    opt = options.parse(remaining_args)
    opt.model_path = args.model_path

    src_fid.slurm.init_distributed_mode(opt)
    src_fid.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    
    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)

    logger = src_fid.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    
    if not directory_exists and opt.is_main:
        options.print_options(opt)
    
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src_fid.data.Collator(tokenizer,
                                              opt.mode,
                                              opt.text_maxlength,
                                              extra_question=opt.extra_question)
        
    eval_examples = src_fid.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src_fid.data.Dataset(eval_examples, opt, is_eval=True)    

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=1,
        collate_fn=collator_function
    )
    
    if args.model_path in ['google/flan-t5-base', 'google/flan-t5-large', 'qinyuany/my-t0-base', 'qinyuany/my-t0-large']:
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(args.model_path)
        model = src_fid.model.FiDT5(t5.config, opt)
        model.load_t5(t5.state_dict())
    else:
        model_class = src_fid.model.FiDT5
        model = model_class.from_pretrained(opt.model_path, opt)

    model = model.to(opt.device)
    
    logger.info("Start eval")
    exactmatch, total, preds = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    evaluation_table = ResultTable(table_name='Eval Result', header=list(preds.keys()))
    evaluation_table.add_row('pred.', preds)

    logger.info(evaluation_table.to_string())

    if opt.write_crossattention_scores:
        src_fid.util.save_distributed_dataset(eval_dataset.data, opt)
