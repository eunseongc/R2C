# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=0)
        self.parser.add_argument('--total_steps', type=int, default=64000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--use_acc', action='store_true', help='')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')
        self.parser.add_argument('--wandb_tag', type=str, default=None)
        self.parser.add_argument('--n_neg_samples', type=int, default=2)

    def add_eval_options(self):
        self.parser.add_argument('--eval_group', type=str, default=None, help='choose a question group to evaluate')
        self.parser.add_argument('--select_gold_context', action='store_true', help='select gold passage and use it only')
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', 
                        help='save dataset with cross-attention scores')
        self.parser.add_argument('--token_scores_path', type=str, default=None, help='path to save token scores (cross-attention scores)')
        self.parser.add_argument('--output_version', type=str, default=None, help='input version')
        self.parser.add_argument('--cut_offs', type=list, default=[1, 2, 3, 5, 10, 20], help='cutoffs for recall')
        self.parser.add_argument('--pseudo_question', type=str, default="")
        self.parser.add_argument('--last_layer_only', type=bool, default=False, help='Use only last decoder layer to calculate token scores')


    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default=None, help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--mode', type=str, default=None, choices=['pair', 'single'], help='pair is original fid')
        self.parser.add_argument('--pretrained_model_path', type=str, default='t5-base')
        self.parser.add_argument('--model_class', type=str, default='FiDT5', help='model class, e.g., FiDT5, FiD_encoder')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--text_maxlength', type=int, default=192, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_contexts', type=int, default=20)
        self.parser.add_argument('--sce_n_contexts', type=int, default=None)
        self.parser.add_argument('--ctx_anno', type=str, default='has_answer', help="e.g., has_answer, mytho")
        self.parser.add_argument('--n_qas', type=int, default=None)
        self.parser.add_argument('--n_extra_contexts', type=int, default=10)
        self.parser.add_argument('--n_max_groups', type=int, default=1)
        self.parser.add_argument('--use_major_group', action='store_true', default=False)
        self.parser.add_argument('--use_max_voting', action='store_true', default=False)
        self.parser.add_argument('--use_max_first_prob', action='store_true', default=False)
        self.parser.add_argument('--use_recursive_graph', action='store_true', default=False)
        self.parser.add_argument('--use_half_extra', action='store_true', default=False)
        self.parser.add_argument('--use_group_dpr', action='store_true', default=False)
        self.parser.add_argument('--ret_path', type=str, default='pretrained_models/nq_retriever')
        self.parser.add_argument('--tokens_k', type=str, default=None, help="e.g., f1, m2, l3, f4")  
        self.parser.add_argument('--replace_bos_token', action='store_true', default=False)
        self.parser.add_argument('--use_local_interaction', action='store_true', default=False)
        self.parser.add_argument('--use_task_weight', action='store_true', default=False)

        

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='eun_test', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_fid', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default=None, help='path for retraining')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local-rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        self.parser.add_argument('--num_workers', type=int, default=10, help="")
        # training parameters
        self.parser.add_argument('--print_freq', type=int, default=2000,
                        help='print training loss every <print_freq> steps')
        self.parser.add_argument('--eval_freq', type=int, default=500,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--eval_from', type=int, default=0)
        self.parser.add_argument('--save_freq', type=int, default=5000,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=1000,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self, args=None):
        if args is None:
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
            
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()