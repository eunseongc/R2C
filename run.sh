python main.py --input_path qa_data/test_20.json \
               --use_token_scores \
               --token_scores_path token_scores/token_scores_list_dpr_20_oneContextFalse.pkl \
               --comp_ctx \
               --ctx_score_cumsum 0.4 \
               --do_sort_ctx \
               --comp_sent \
               --sent_low 0.7