import re
import tiktoken
import numpy as np

from collections import Counter
from nltk import sent_tokenize
from IPython import embed

GPT_TOKENIZER = tiktoken.encoding_for_model("gpt-3.5-turbo")

def gini(x):
    mean_absolute_diff = np.abs(np.subtract.outer(x, x)).mean()
    relative_mean_absolute_diff = mean_absolute_diff/np.mean(x)
    g = 0.5 * relative_mean_absolute_diff
    return g


def get_ctx_start_idx(batch_token_ids, tokenizer, pattern_str):
    pattern_token_ids = tokenizer.encode(pattern_str, add_special_tokens=False)
    len_pattern = len(pattern_token_ids)    
    start_indices = np.where(batch_token_ids == pattern_token_ids[0])[1]
    if len(start_indices) == len(batch_token_ids) and len(np.unique(start_indices)) == 1:
        ## Patterns are all in the same position
        ctx_start_idx = start_indices[0] + len_pattern ## Start index of the context
    else:
        indices_counter = Counter(start_indices)
        ## Only remain index that are same for all the passages
        remove_indices = set()
        for start_index, _ in indices_counter.items():
            if not np.all(batch_token_ids[:, start_index:start_index+len_pattern] == pattern_token_ids):
                remove_indices.add(start_index)
        indices_counter = {k: v for k, v in indices_counter.items() if k not in remove_indices and v == len(batch_token_ids)}

        if len(indices_counter) == 1:
            ctx_start_idx = list(indices_counter.keys())[0] + len_pattern
        else:
            ## Get the last one
            ctx_start_idx = list(indices_counter.keys())[-1] + len_pattern

    return ctx_start_idx


def get_ctx_scores(batch_scores, batch_token_ids, mode: str, question_mode: str, include_end_token: bool, tokenizer, pattern_str):
    ## Only question scoring 추가
    if question_mode == 'include':
        batch_scores_ctxs = batch_scores
    elif question_mode == 'exclude':
        ctx_start_idx = get_ctx_start_idx(batch_token_ids, tokenizer, pattern_str)
        batch_scores_ctxs = batch_scores[:, ctx_start_idx:]
    elif question_mode =='only':
        ctx_start_idx = get_ctx_start_idx(batch_token_ids, tokenizer, pattern_str)
        pattern_token_ids = tokenizer.encode(pattern_str, add_special_tokens=False)
        len_pattern = len(pattern_token_ids) 
        batch_scores_ctxs = batch_scores[:, :ctx_start_idx - len_pattern]
        
    if include_end_token:
        end = None
    else:
        end = -1

    ctx_scores = []
    for scores in batch_scores_ctxs:
        if mode == 'mean':
            ctx_scores.append(scores[scores != 0][:end].mean())
        elif mode == 'max':
            ctx_scores.append(scores[scores != 0][:end].max())
        elif mode == 'sum': ## Not fair as it is based on the length of the context
            ctx_scores.append(scores[scores != 0][:end].sum())
        else:
            raise ValueError(f"Invalid mode: {mode}")

    norm_ctx_scores = np.array(ctx_scores) / np.sum(ctx_scores)
    return norm_ctx_scores


def compress_contexts(args, batch_scores, batch_token_ids, tokenizer, ctxs, ctx_comp_len: int, pattern_str: str):
    ############################################
    ### Context compression using FiD score ###
    ############################################

    do_sort_ctx=args.do_sort_ctx
    ctx_score_mode=args.ctx_score_mode
    question_mode=args.question_mode
    include_end_token=args.include_end_token
    
    norm_ctx_scores = get_ctx_scores(batch_scores, batch_token_ids, ctx_score_mode, question_mode, include_end_token, tokenizer, pattern_str)
    ctx_indices_sorted = np.argsort(norm_ctx_scores)[::-1].tolist()
    len_total = 0
    ctx_indices = []
    for idx in ctx_indices_sorted:
        if 'longbench' in args.dataset:
            len_total += len(GPT_TOKENIZER.encode(ctxs[idx]['text']))
        else:
            len_total += len(GPT_TOKENIZER.encode(f"Document [{idx}](Title: {ctxs[idx]['title']}) {ctxs[idx]['text']}"))

        if len_total >= ctx_comp_len:
            break
        ctx_indices.append(idx)

    ## By score percentile
    # cum_ctx_index = np.where(norm_ctx_scores[norm_ctx_scores.argsort()[::-1]].cumsum() > ctx_comp_tokens)[0][0] + 1
    # ctx_indices_selected = np.argsort(norm_ctx_scores)[::-1][:cum_ctx_index]

    if do_sort_ctx:
        ## Remain sorted order
        ctx_indices_selected = ctx_indices
    else:
        ## Restore the original order
        ctx_indices_selected = sorted(ctx_indices)

    ctxs = [ctxs[i] for i in ctx_indices_selected]
    batch_scores_selected = batch_scores[ctx_indices_selected] ## result shape: (cut_off, max_len)
    batch_token_ids_selected = batch_token_ids[ctx_indices_selected] ## result shape: (cut_off, max_len)

    return batch_scores_selected, batch_token_ids_selected, ctxs, ctx_indices


def compress_sentences(batch_scores, batch_token_ids, tokenizer, ctxs, ctx_indices_sorted, sent_comp_len: int, adaptive_sent_comp: bool, pattern_str: str):
    ############################################
    ### Sentence compression using FiD score ###
    ############################################
    # Preparing lists to store the results
    
    ## Decode 해서 더해줄 것이 아니라, 점수로... 어떻게 해야할듯..?
    ## [ES] batch_len_context is not used
    ctx_start_idx = get_ctx_start_idx(batch_token_ids, tokenizer, pattern_str)
    batch_eos_token_idx, batch_len_context = [], []
    for token_ids in batch_token_ids:
        eos_token_idx = np.where(token_ids == 1)[0][-1]
        batch_eos_token_idx.append(eos_token_idx)
        batch_len_context.append(eos_token_idx - ctx_start_idx)

    batch_scores_context = [batch_scores[i][ctx_start_idx:eos_token_idx] for i, eos_token_idx in enumerate(batch_eos_token_idx)]
    batch_token_ids_context = [batch_token_ids[i][ctx_start_idx:eos_token_idx] for i, eos_token_idx in enumerate(batch_eos_token_idx)]

    sents_list = [sent_tokenize(ctx['text']) for ctx in ctxs]
    split_token_ctxs = []
    for ctx_i, ctx in enumerate(ctxs):
        split_token_ctx = ['']
        for sent in sents_list[ctx_i][1:]:
            split_token = ctx['text'][ctx['text'].find(sent) - 1]
            split_token_ctx.append(split_token)
        split_token_ctxs.append(split_token_ctx)

    sent_mean_score_ctxs = []
    sent_scores_ctxs = []
    sent_token_ids_ctxs = []

    for context_score, context_ids, sents in zip(batch_scores_context, batch_token_ids_context, sents_list):
        try:
            sent_len_list = [len(tokens_sent) for tokens_sent in tokenizer.batch_encode_plus(sents, add_special_tokens=False)['input_ids']]
        except:
            embed()

        cum_len_list = [sum(sent_len_list[:i+1]) for i in range(len(sent_len_list))]
        start_idx = 0
        sent_mean_score_ctx = []
        sent_scores_ctx = []
        sent_token_idx_ctx = []
        for cum_len in cum_len_list:
            if start_idx >= context_score.shape[0]:
                continue
            sent_mean_score_ctx.append(np.mean(context_score[start_idx:cum_len]))
            sent_scores_ctx.append(context_score[start_idx:cum_len])
            sent_token_idx_ctx.append(context_ids[start_idx:cum_len])
            ## Update     
            start_idx = cum_len

        sent_mean_score_ctxs.append(np.array(sent_mean_score_ctx))
        sent_scores_ctxs.append(sent_scores_ctx)
        sent_token_ids_ctxs.append(sent_token_idx_ctx)

    num_ctxs = len(ctxs)
    if adaptive_sent_comp:
        ## Adaptive sentence compression
        ## sum should be the sent_comp_len
        d = 2 * sent_comp_len / (num_ctxs * (num_ctxs - 1))
        comp_len_per_ctx = [int(d * i) for i in range(num_ctxs)]
    else:
        comp_len_per_ctx = [int(sent_comp_len / num_ctxs) for _ in range(num_ctxs)]
        
    new_token_ids_ctxs = []
    new_scores_ctxs = []
    rank_map = {ctx_i: rank for rank, ctx_i in enumerate(ctx_indices_sorted)}
    cur_ctx_indices = [ctx['org_idx'] - 1 for ctx in ctxs]
    total_comp_len = 0
    # for i, idx in enumerate(cur_ctx_indices):
    for i, idx in reversed(list(enumerate(cur_ctx_indices))):
        sent_mean_score_ctx = sent_mean_score_ctxs[i]
        sents = sents_list[i]
        if len(sent_mean_score_ctx) != len(sents):
            ## When the length of the question + context was longer than the max length of the FiD, this can happen
            ## In this case, we should match sents to sent_mean_score_ctx
            sents = sents[:len(sent_mean_score_ctx)]
            
        rank = rank_map[idx]
        comp_len = comp_len_per_ctx[rank]
        
        argsort_sent = np.argsort(sent_mean_score_ctx) ## Descending order
        cur_comp_len = 0

        if total_comp_len >= sent_comp_len or comp_len == 0:
            r = 0
        else:
            for r, sent_index in enumerate(argsort_sent):
                cur_sent_len = len(GPT_TOKENIZER.encode(sents[sent_index]))
                cur_comp_len += cur_sent_len
                total_comp_len += cur_sent_len
                if total_comp_len >= sent_comp_len or cur_comp_len >= comp_len:
                    break
        
            ### at last 1 sentence should be included
            if r == len(sents) - 1:
                if rank == 0:
                    pass
                else:
                    cur_comp_len = cur_comp_len - cur_sent_len ## We don't want to include the last sentence
                    total_comp_len -= cur_sent_len
                    ## Increase the comp_len of lower ranked context
                    comp_len_per_ctx[rank-1] += comp_len - cur_comp_len
            else:
                r = r + 1
        # print(comp_len_per_ctx)

        argsort_sent = argsort_sent[r:]
        if len(argsort_sent) == 0:
            from IPython import embed; embed()
            raise ValueError("No sentence is selected, at least one sentence should be selected.")
        else:
            sent_indices_selected = np.sort(argsort_sent)
            sent_comp_text = ""
            for sent_index in sent_indices_selected:
                sent_comp_text += split_token_ctxs[i][sent_index] + sents[sent_index]
            ctxs[i]['text'] = sent_comp_text

            new_scores_ctx = np.concatenate([sent_scores_ctxs[i][sent_index] for sent_index in sent_indices_selected])
            new_scores_ctxs.append(new_scores_ctx)
            new_token_ids_ctx = np.concatenate([sent_token_ids_ctxs[i][sent_index] for sent_index in sent_indices_selected])
            new_token_ids_ctxs.append(new_token_ids_ctx)

    # grad_comp_ratio_sent = np.linspace(sent_low, sent_high, len(ctxs ))[::-1]
    ## Using percentile
    # for ctx_i, (sent_mean_score_ctx, comp_ratio) in enumerate(zip(sent_mean_score_ctxs, grad_comp_ratio_sent)):
    #     ## Select sent based on sent_mean_score_ctx up to comp_ratio (stop if over comp_ratio)
    #     sent_indices_selected = []
    #     cum_score = 0
    #     sent_mean_score_ctx_norm_sorted = np.sort(sent_mean_score_ctx/sent_mean_score_ctx.sum())[::-1]
    #     indices_sorted = np.argsort(sent_mean_score_ctx)[::-1]
    #     for idx, sent_score in enumerate(sent_mean_score_ctx_norm_sorted):
    #         cum_score += sent_score
    #         sent_indices_selected.append(indices_sorted[idx])
    #         if cum_score > comp_ratio:
    #             break
    #     sent_indices_selected = np.sort(sent_indices_selected)
    #     sent_comp_text = ""
    #     for i in sent_indices_selected:
    #         sent_comp_text += split_token_ctxs[ctx_i][i] + sents_list[ctx_i][i]
    #     # sent_comp_text = ' '.join([sents_list[ctx_i][i] for i in sent_indices_selected])
    #     ctxs[ctx_i]['text'] = sent_comp_text

    #     new_scores_ctx = np.concatenate([sent_token_ids_ctxs[ctx_i][i] for i in sent_indices_selected])
    #     new_scores_ctxs.append(new_scores_ctx)
    #     new_token_ids_ctx = np.concatenate([sent_token_ids_ctxs[ctx_i][i] for i in sent_indices_selected])
    #     new_token_ids_ctxs.append(new_token_ids_ctx)
    return new_scores_ctxs, new_token_ids_ctxs, ctxs


def compress_tokens(batch_scores, batch_token_ids, ctxs, token_lamb, tokenizer):
    ############################################
    ### 3. Token compression using FiD score ###
    ############################################
    len_ctxs = [ctx_scores.shape[0] for ctx_scores in batch_scores]

    ## Concatenate all the scores and ids in the batch
    batch_scores = np.concatenate(batch_scores)
    batch_token_ids = np.concatenate(batch_token_ids)

    # target_len = max(int(batch_scores.shape[0] * token_lamb), 100) ## Not cur_len, as it is the length based on chatgpt_tok
    target_len = int(batch_scores.shape[0] * token_lamb)

    cum_len_list = [sum(len_ctxs[:i+1]) for i in range(len(len_ctxs))]
    sorted_target_indices = np.sort(batch_scores.argsort()[::-1][:target_len])

    start = 0
    for cum_len, ctx in zip(cum_len_list, ctxs):
        end = cum_len
        cur_target_indices = sorted_target_indices[(sorted_target_indices >= start) & (sorted_target_indices < end)]
        empty_token_indices = np.where(batch_token_ids[start:end] == 3)[0] + start
        cur_target_indices = np.sort(np.union1d(cur_target_indices, empty_token_indices))
        new_batch_token_ids = batch_token_ids[cur_target_indices]
        new_batch_token_ids_filtered = new_batch_token_ids[new_batch_token_ids != 2]
        tok_comp_text = tokenizer.decode(new_batch_token_ids_filtered)
        tok_comp_text = re.sub(r' {2,}', ' ', tok_comp_text)
        ctx['text'] = tok_comp_text
        start = end

    ## [ES] Should I return batch_token_scores, batch_token_ids too?
    return ctxs



'''

span_em_list = []
for pred in preds:
    cnt = pred['model_prompt'].count('Document [')
    gold_answers = pred["answers"]
    model_answer = pred["model_answer"]
    span_em_list.append(best_subspan_em(prediction=model_answer, ground_truths=gold_answers))
print(f'{100 * np.mean(span_em_list):.2f}')




3.86
66.71
4.46
67.07
6.96
66.81
7.29
66.55

3.88
51.76
4.47
51.50
6.96
51.15
7.27
50.32

0.05 (문서 압축을 많이한 것)
total: 58.89 low_indices: 66.71 / high_indices: 51.76

0.1
total: 58.92 low_indices: 67.07 / high_indices: 51.50

0.15
total: 58.65 low_indices: 66.64 / high_indices: 51.36

0.2
total: 58.84 low_indices: 66.98 / high_indices: 51.43

0.25

0.3
total: 58.62 low_indices: 66.81 / high_indices: 51.15

0.35 (문서 압축을 조금한 것)
total: 58.06 low_indices: 66.55 / high_indices: 50.32



3.87
58.89
51.76
66.71
4.47
58.92
51.50
67.07
5.06
58.65
51.36
66.64
5.90
58.84
51.43
66.98





for rate in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
for rate in [0.05, 0.3]:
for rate in [0.05, 0.1, 0.15, 0.2]:
    with xopen(f'fid_500_ctxTrue_sentTrue{rate}_tokFalse1.0_orgidxTrue_Llama-2-13b-chat-hf.jsonl.gz') as f:
        preds = [json.loads(l) for l in f]
        doc_num_list = []
        span_em_list = []
        span_em_list_low = []
        span_em_list_high = []

        for p_i, pred in enumerate(preds):
            cnt = pred['model_prompt'].count('Document [')
            doc_num_list.append(cnt)
            gold_answers = pred["answers"]
            model_answer = pred["model_answer"]
            span_em = best_subspan_em(prediction=model_answer, ground_truths=gold_answers)
            span_em_list.append(span_em)
            if p_i in low_indices:
                span_em_list_low.append(span_em)
            else:
                span_em_list_high.append(span_em)
        print(f'{np.mean(doc_num_list):.2f}')
        print(f'{100 * np.mean(span_em_list):.2f}')
        print(f'{100 * np.mean(span_em_list_low):.2f}')
        print(f'{100 * np.mean(span_em_list_high):.2f}')
        
doc_num_list = []
for d in data:
    cnt = d['compressed_prompt'].count('Document [')
    doc_num_list.append(cnt)
print(f'{np.mean(doc_num_list):.2f}')

'''