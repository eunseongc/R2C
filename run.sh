for ctx_score_cumsum in 0.3
do
    for sent_low in 0.6 0.7 0.8
    do
        python main.py --input_path qa_data/test_20.json \
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


for ctx_score_cumsum in 0.4
do
    for sent_low in 0.2 0.3 0.4 0.5 
    do
        python main.py --input_path qa_data/test_20.json \
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