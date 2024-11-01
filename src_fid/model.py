# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import torch.sparse as sparse
import transformers
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from .t5_model import CustomT5ForConditionalGeneration


class FiDT5(CustomT5ForConditionalGeneration):
    def __init__(self, config, opt=None):
        super().__init__(config, opt)
        self.opt = opt

        self.wrap_encoder(opt)
        self.sep_q_p = opt.sep_q_p
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(opt.pretrained_model_path)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        if kwargs.get('step') is not None:
            del kwargs['step']

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs 

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_new_tokens, return_dict_in_generate=False, bsz=250, last_layer_only=False, **kwargs):

        ##############################
        ### Encoding with the batch
        self.encoder.n_passages = input_ids.size(1)
        encoder_outputs = self.encoder(input_ids=input_ids.view(input_ids.size(0), -1),
                                                                             attention_mask=attention_mask.view(attention_mask.size(0), -1),
                                                                             **kwargs)
        ##############################

        ##############################
        ### Encoding with the minibatches
        # encoder_outputs_list, attn_mask_list, probs_list = [], [], []
        # for i in range(0, input_ids.size(1), bsz):
        #     size = min(bsz, input_ids.size(1) - i)
        #     self.encoder.n_passages = size
        #     encoder_outputs, _, _, probs = self.encoder(input_ids=input_ids[0, i:i+size].view(input_ids.size(0), -1),
        #                                                                      attention_mask=attention_mask[0, i:i+size].view(input_ids.size(0), -1),
        #                                                                      **kwargs)
        #     encoder_outputs_list.append(encoder_outputs.last_hidden_state)
        #     attn_mask_list.append(encoder_outputs.attention_mask)
        #     probs_list.append(probs)
        # encoder_outputs.last_hidden_state = torch.cat(encoder_outputs_list, dim=1)
        # encoder_outputs.attention_mask = torch.cat(attn_mask_list, dim=1)
        # probs = torch.cat(probs_list, dim=1)
        ##############################

        self.encoder.n_passages = input_ids.size(1)
        kwargs['return_dict_in_generate'] = return_dict_in_generate
        
        outputs = super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        if return_dict_in_generate:
            cross_attentions = outputs.cross_attentions
            outputs = outputs.sequences
            if kwargs['output_attentions']:
                crossattention_scores, token_scores = self.get_crossattention_scores_my(outputs, cross_attentions, attention_mask, last_layer_only=last_layer_only)
                ## Crossattention scores vs. token scores
                return outputs, crossattention_scores, token_scores
        
        return outputs

    def wrap_encoder(self, opt, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, opt=opt, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(self.opt)

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores_my(self, sequences, cross_attentions, attention_mask, last_layer_only=False):
        """
        sequences: torch.tensor (bsz, #gen tokens)
        cross_attentions: list(#gen tokens) of list(#layers) of (bsz, n_heads, 1, n_passages * text_maxlength)
        attention_mask: torch.tensor (bsz, n_passages, text_maxlength)
        """
        
        # Assuming that the cross_attentions are arranged as a list of [gen tokens][layers], where each element is
        # a tensor of shape (bsz, n_heads, 1, n_passages * text_maxlength)
        
        cross_attentions_per_gen_token = [torch.stack(gen_tok_cross_attention) for gen_tok_cross_attention in cross_attentions]
        cross_attention_scores_all = torch.stack(cross_attentions_per_gen_token)
        
        bsz, n_passages,text_maxlength = attention_mask.size()
        n_gen_tokens, n_layers, bsz, n_heads, n_seq, _ = cross_attention_scores_all.size()
        n_heads = cross_attention_scores_all.size(3)
        n_layers = cross_attention_scores_all.size(1)

        cross_attention_scores_all = cross_attention_scores_all[0]

        if last_layer_only:
            cross_attention_scores_all = cross_attention_scores_all[-1, None]
            n_layers = 1


        scores = cross_attention_scores_all.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~attention_mask[:, None, None], 0.)
        token_scores = scores.sum(dim=[1,2]).squeeze(0).tolist() ## Squeeze for only batch. (use if only when you are using 1 passage? why?)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = attention_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens

        return scores, token_scores

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

class EncoderWrapper(nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, opt=None, use_checkpoint=False):
        super().__init__()

        self.main_input_name = 'input_ids'
        self.encoder = encoder
        self.config = encoder.config
        self.d_model = self.config.d_model
        ## Parameters from options
        self.sep_q_p = opt.sep_q_p
        self.extra_question = opt.extra_question
        self.use_local_interaction = opt.use_local_interaction
        self.tokens_k = opt.tokens_k

        self.opt = opt
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    ## True encoder 
    def forward(self, input_ids=None, attention_mask=None, q_tokens=None, has_answers=None, **kwargs):
        # total_length = n_passages * passage_length

        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        n_passages = self.n_passages

        outputs.last_hidden_state = outputs.last_hidden_state.contiguous().view(bsz, n_passages*passage_length, -1)
        attention_mask = attention_mask.contiguous().view(bsz, n_passages*passage_length)
        hidden_states = outputs.last_hidden_state

        outputs.last_hidden_state = hidden_states

        outputs.attention_mask = attention_mask.contiguous().view(bsz, -1)
        return outputs

class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


class FiD_classifier(transformers.BertForSequenceClassification):
    def __init__(self, config, opt=None):
        super().__init__(config)
        self.opt = opt
        
        self.score_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_act = nn.GELU()
        self.score_layer2 = nn.Linear(config.hidden_size, 1)
        self.score_layer = nn.Sequential(self.score_layer1, self.proj_act, self.score_layer2)

        nn.init.normal_(self.score_layer1.weight, std=0.02)
        nn.init.zeros_(self.score_layer1.bias)
        nn.init.normal_(self.score_layer2.weight, std=0.02)
        nn.init.zeros_(self.score_layer2.bias)

    def forward(self, **batch):
        outputs = super().forward(**batch)
        new_logits = self.score_layer(outputs.pooled_output)
        outputs.logits = new_logits
        return outputs
    
class FiD_encoder(transformers.T5EncoderModel):
    def __init__(self, config, opt=None):
        super().__init__(config)
        self.opt = opt
        self.wrap_encoder(opt)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_ids=None, attention_mask=None):
        bsz, n_passages, passage_length = input_ids.shape
        input_ids = input_ids.view(bsz*n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*n_passages, passage_length)
        outputs = self.encoder.encoder(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        return outputs 

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(self.opt)

    def wrap_encoder(self, opt, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, opt=opt, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint