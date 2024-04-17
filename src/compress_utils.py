import re
import numpy as np

from nltk import sent_tokenize

def compress_contexts(batch_scores, batch_token_ids, ctx_score_cumsum: float, do_sort: bool):
    ############################################
    ### Document compression using FiD score ###
    ############################################
    doc_scores = []
    for scores in batch_scores:
        doc_scores.append(scores[scores != 0].mean())
    doc_scores = np.array(doc_scores) / np.sum(doc_scores)

    cum_doc_index = np.where(doc_scores[doc_scores.argsort()[::-1]].cumsum() > ctx_score_cumsum)[0][0] + 1
    doc_indices_selected = np.argsort(doc_scores)[::-1][:cum_doc_index]
    batch_scores = batch_scores[doc_indices_selected] ## result shape: (cut_off, max_len)
    batch_token_ids = batch_token_ids[doc_indices_selected] ## result shape: (cut_off, max_len)

    ## [ES] batch_len_context is not used
    batch_context_start_idx, batch_eos_token_idx, batch_len_context = [], [], []
    for token_ids in batch_token_ids:
        context_start_idx = np.where(token_ids == 2625)[0][0] + 2    
        eos_token_idx = np.where(token_ids == 1)[0][0]
        batch_context_start_idx.append(context_start_idx)
        batch_eos_token_idx.append(eos_token_idx)
        batch_len_context.append(eos_token_idx - context_start_idx)
        
    batch_scores_selected = [batch_scores[i][context_start_idx:eos_token_idx] for i, (context_start_idx, eos_token_idx) in enumerate(zip(batch_context_start_idx, batch_eos_token_idx))]
    batch_token_ids_selected = [batch_token_ids[i][context_start_idx:eos_token_idx] for i, (context_start_idx, eos_token_idx) in enumerate(zip(batch_context_start_idx, batch_eos_token_idx))]

    return batch_scores_selected, batch_token_ids_selected, doc_indices_selected, 


def compress_sentences(batch_scores, batch_token_ids, ctxs, sent_low, sent_high, tokenizer):
    ############################################
    ### Sentence compression using FiD score ###
    ############################################
    # Preparing lists to store the results
    sent_mean_score_ctxs = []
    sent_scores_ctxs = []
    sent_token_ids_ctxs = []

    sents_list = [sent_tokenize(ctx['text']) for ctx in ctxs]
    for context_score, context_ids, sents in zip(batch_scores, batch_token_ids, sents_list):
        sent_len_list = [len(tokens_sent) for tokens_sent in tokenizer.batch_encode_plus(sents, add_special_tokens=False)['input_ids']]
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
        
    new_token_ids_ctxs = []
    new_scores_ctxs = []
    grad_comp_ratio_sent = np.linspace(sent_low, sent_high, len(ctxs ))[::-1]
    for ctx_i, (sent_mean_score_ctx, comp_ratio) in enumerate(zip(sent_mean_score_ctxs, grad_comp_ratio_sent)):
        ## Select sent based on sent_mean_score_ctx up to comp_ratio (stop if over comp_ratio)
        sent_indices_selected = []
        cum_score = 0
        sent_mean_score_ctx_norm_sorted = np.sort(sent_mean_score_ctx/sent_mean_score_ctx.sum())[::-1]
        indices_sorted = np.argsort(sent_mean_score_ctx)[::-1]
        for idx, sent_score in enumerate(sent_mean_score_ctx_norm_sorted):
            cum_score += sent_score
            sent_indices_selected.append(indices_sorted[idx])
            if cum_score > comp_ratio:
                break
        sent_indices_selected = np.sort(sent_indices_selected)
        sent_comp_text = ' '.join([sents_list[ctx_i][i] for i in sent_indices_selected])
        ctxs[ctx_i]['text'] = sent_comp_text

        new_scores_ctx = np.concatenate([sent_token_ids_ctxs[ctx_i][i] for i in sent_indices_selected])
        new_scores_ctxs.append(new_scores_ctx)
        new_token_ids_ctx = np.concatenate([sent_token_ids_ctxs[ctx_i][i] for i in sent_indices_selected])
        new_token_ids_ctxs.append(new_token_ids_ctx)

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

# sent_high = 1.0
# sent_low_list = [0.3]
# token_lamb_list = [1.0]

# for doc_score_cumsum in doc_score_cumsum_list:
#     for sent_low in sent_low_list:
#         for token_lamb in token_lamb_list:
#             len_changes = []
#             ctxs_len_list = []
#             for qas_i, (batch_scores, batch_token_ids) in enumerate(tqdm(token_scores_list_org)):
                
#                 #####################
#                 qas = test[qas_i]
#                 ctxs = qas['ctxs']
#                 sent_mean_score_ctxs = []
#                 sent_scores_ctxs = []
#                 sent_token_ids_ctxs = []

#                 doc_scores = []
#                 for scores in np.array(batch_scores):
#                     doc_scores.append(scores[scores != 0].mean())
#                 doc_scores = np.array(doc_scores) / np.sum(doc_scores)

#                 sents_list = [sent_tokenize(ctx['text']) for ctx in ctxs]
#                 for context_score, context_ids, sents in zip(np.array(batch_scores), batch_token_ids.squeeze().numpy(), sents_list):
#                     sent_len_list = [len(tokens_sent) for tokens_sent in t5_tok.batch_encode_plus(sents, add_special_tokens=False)['input_ids']]
#                     cum_len_list = [sum(sent_len_list[:i+1]) for i in range(len(sent_len_list))]
#                     start_idx = 0
#                     sent_mean_score_ctx = []
#                     sent_scores_ctx = []
#                     sent_token_idx_ctx = []
#                     for cum_len in cum_len_list:
#                         if start_idx >= context_score.shape[0]:
#                             continue
#                         sent_mean_score_ctx.append(np.mean(context_score[start_idx:cum_len]))
#                         sent_scores_ctx.append(context_score[start_idx:cum_len])
#                         sent_token_idx_ctx.append(context_ids[start_idx:cum_len])
#                         ## Update     
#                         start_idx = cum_len
                    
#                     sent_mean_score_ctxs.append(np.array(sent_mean_score_ctx))
#                     sent_scores_ctxs.append(sent_scores_ctx)
#                     sent_token_ids_ctxs.append(sent_token_idx_ctx)

#                 d_g, s_g, d_e, s_e = gini(doc_scores), gini(np.sort(np.concatenate(sent_mean_score_ctxs)/np.sum(np.concatenate(sent_mean_score_ctxs)))), entropy(doc_scores), entropy(np.sort(np.concatenate(sent_mean_score_ctxs)/np.sum(np.concatenate(sent_mean_score_ctxs))))
#                 # if d_g < 0.20811377566121111:
#                 if d_g < 0.2332994939256315:
#                     doc_score_cumsum = 0.3
#                     sent_low = 0.7
#                 else:
#                     doc_score_cumsum = 0.4
#                     sent_low = 0.5
                
#                 # # if s_g < 0.6489349705185483:
#                 #     # sent_low =     
#                 # #############################
                
                
#                 len_change = []
#                 qas = test[qas_i]
                
#                 batch_scores = np.array(batch_scores) ## result shape: (20, max_len)
#                 batch_token_ids = batch_token_ids.squeeze().numpy() ## result shape: (20, max_len)

#                 ## Current length
#                 text_list = [f"Document [{ctx['org_idx']}](Title: {ctx['title']}) {ctx['text']}" for ctx in qas['ctxs']]
#                 org_prompt = INSTRUCTION + '\n\n' + '\n'.join(text_list) + '\n\n' + QUESTION_TEMPLATE.format(qas['question'])
                
#                 len_change.append(len(chatgpt_tok.encode(org_prompt)))

                
                


#             doc_len_list, sent_len_list, tok_len_list = [], [], []
#             for len_change in len_changes:
#                 _, doc, sent, tok = len_change
#                 doc_len_list.append(doc)
#                 sent_len_list.append(sent)
#                 tok_len_list.append(tok)
#             print(f"[doc_score_cumsum: {doc_score_cumsum}, sent_low: {sent_low}, sent_high: {sent_high}, token_lamb: {token_lamb}] Org. len: {int(org_avg_len)}, doc_comp: {int(np.mean(doc_len_list))} ({100 * np.mean(doc_len_list)/org_avg_len:.2f}%, avg #: {np.mean(ctxs_len_list):.1f}), sent_comp: {int(np.mean(sent_len_list))} ({100 * np.mean(sent_len_list)/org_avg_len:.2f}%), tok_comp: {int(np.mean(tok_len_list))} ({100 * np.mean(tok_len_list)/org_avg_len:.2f}%)")

#             # with xopen(f'../lost-in-the-middle/qa_data/20_total_documents/dpr_20_0407/nq-open-dpr_20_fid_doc0.0_sl0.0_sh1.0_tl{token_lamb}_0.70.5.jsonl.gz', 'wt') as f:
#             with xopen(f'../lost-in-the-middle/qa_data/20_total_documents/nq-open-20_total_documents_fid_doc0.0_sl0.0_sh1.0_tl{token_lamb}_0.70.5.jsonl.gz', 'wt') as f:
#             # with xopen(f'../lost-in-the-middle/qa_data/20_total_documents/dpr_20_0409/nq-open-dpr_20_fid_doc{doc_score_cumsum}_sl{sent_low}_sh{sent_high}_tl{token_lamb}.jsonl.gz', 'wt') as f:
#                 for qas in test:
#                     f.write(json.dumps(qas) + '\n')
