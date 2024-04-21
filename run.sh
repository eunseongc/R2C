# for ctx_score_cumsum in 0.3
# do
#     for sent_low in 0.6 0.7 0.8
#     do
#         python main.py --input_path qa_data/20_gold_at_0.jsonl.gz \
#                        --output_root compressed_qa_data/20_gold_at_0 \
#                        --use_token_scores \
#                        --token_scores_path token_scores/token_scores_list_20_documents_gold_at_0_oneContextFalse.pkl \
#                        --comp_ctx \
#                        --ctx_score_cumsum ${ctx_score_cumsum} \
#                        --do_sort_ctx \
#                        --comp_sent \
#                        --sent_low ${sent_low}
#     done
# done

# Original avg_len: 2949.10 comp_ctx (0.3): 503.34 comp_sent(0.6-1.0): 352.62
# Original avg_len: 2949.10 comp_ctx (0.3): 503.34 comp_sent(0.7-1.0): 378.24
# Original avg_len: 2949.10 comp_ctx (0.3): 503.34 comp_sent(0.8-1.0): 410.65

for ctx_score_cumsum in 0.4
do
    for sent_low in 0.2 0.3 0.4 0.5 
    do
        python main.py --input_path qa_data/20_gold_at_0.jsonl.gz \
                       --output_root compressed_qa_data/20_gold_at_0 \
                       --use_token_scores \
                       --token_scores_path token_scores/token_scores_list_20_documents_gold_at_0_oneContextFalse.pkl \
                       --comp_ctx \
                       --ctx_score_cumsum ${ctx_score_cumsum} \
                       --do_sort_ctx \
                       --comp_sent \
                       --sent_low ${sent_low}
    done
done

# Original avg_len: 2949.10 comp_ctx (0.4): 743.10 comp_sent(0.2-1.0): 407.10
# Original avg_len: 2949.10 comp_ctx (0.4): 743.10 comp_sent(0.3-1.0): 427.13
# Original avg_len: 2949.10 comp_ctx (0.4): 743.10 comp_sent(0.4-1.0): 454.66
# Original avg_len: 2949.10 comp_ctx (0.4): 743.10 comp_sent(0.5-1.0): 485.33