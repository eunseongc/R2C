dataset=longbench_repobench-p
input_file="longbench_repobench-p.json"
token_scores_file="token_scores_list_repobench-p_0423_oneContextFalse.pkl"

for ctx_score_cumsum in 0.3
do
    python main.py --dataset ${dataset} \
                   --input_path qa_data/0423/${input_file} \
                   --output_root compressed_qa_data/0423 \
                   --use_token_scores \
                   --token_scores_path token_scores/${token_scores_file} \
                   --comp_ctx \
                   --ctx_score_cumsum ${ctx_score_cumsum} \
                   --comp_sent \
                   --sent_low 0.8
done

#  f"fid_gini{args.ctx_gini_standard}_ctx{args.ctx_score_cumsum_gini_low}-{args.ctx_score_cumsum_gini_high}_sent{args.sent_low_gini_high}-{args.sent_low_gini_low}.jsonl.gz"


# for ctx_gini_standard in 0.05 0.15
# do
#     python main.py --input_path qa_data/${input_file} \
#                    --output_root compressed_qa_data/${data} \
#                    --use_token_scores \
#                    --token_scores_path token_scores/${token_scores_file} \
#                    --use_gini \
#                    --ctx_gini_standard ${ctx_gini_standard} \
#                    --ctx_score_cumsum_gini_low 0.3 \
#                    --ctx_score_cumsum_gini_high 0.4 \
#                    --sent_low_gini_high 0.8 \
#                    --sent_low_gini_low 0.3 \
#                    --comp_ctx \
#                    --do_sort_ctx \
#                    --comp_sent
# done
