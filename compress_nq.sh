## compress_nq bash 예시
# source run_nq.sh test3610 500 500_ours-base token_scores_list_nq_dpr_test3610_20_ours-base.pkl &
# source run_nq.sh test3610 500 500_ours-large token_scores_list_nq_dpr_test3610_20_ours-large.pkl &
# source run_nq.sh test3610 500 500_flan-t5-base token_scores_list_nq_dpr_test3610_20_flan-t5-base.pkl &
# source run_nq.sh test3610 500 500_flan-t5-large token_scores_list_nq_dpr_test3610_20_flan-t5-large.pkl &
# source run_nq.sh test3610 500 500_t0-base token_scores_list_nq_dpr_test3610_20_t0-base.pkl &
# source run_nq.sh test3610 500 500_t0-large token_scores_list_nq_dpr_test3610_20_t0-large.pkl

split=$1
target_length=$2
output_version=$3
token_scores_file=$4
dataset="nq_dpr_${split}"

# model_size=$2
# token_scores_file="${model_size}_token_scores_list_${dataset}_20_t5-base_empty.pkl"
# token_scores_file="${model_size}_token_scores_list_${dataset}_20_t5-base_summarize.pkl"
# token_scores_file="${model_size}_token_scores_list_${dataset}_20_t5-base_random.pkl"

# token_scores_file="base_token_scores_list_${dataset}_20_ours_512.pkl" ## R2C paper report one

# token_scores_file="${model_size}_token_scores_list_${dataset}_20_ours_last_layer.pkl"
# token_scores_file="${model_size}_token_scores_list_${dataset}_20_t5-base.pkl"



# ### chunk X sentence X tok O
# python compress.py --dataset ${dataset} \
#                    --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                    --output_root data_compressed/${model_size}/${output_version} \
#                    --use_token_scores \
#                    --token_scores_path token_scores/${token_scores_file} \
#                    --target_length ${target_length} \
#                    --use_org_idx \
#                    --question_mode include \
#                    --comp_tok \
#                    --e_tok 100 &

# ### chunk X sentence X tok O sort O
# python compress.py --dataset ${dataset} \
#                    --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                    --output_root data_compressed/${model_size}/${output_version} \
#                    --use_token_scores \
#                    --token_scores_path token_scores/${token_scores_file} \
#                    --target_length ${target_length} \
#                    --do_sort_ctx \
#                    --use_org_idx \
#                    --question_mode include \
#                    --comp_tok \
#                    --e_tok 100 &

# ### chunk X sentence O tok X
# python compress.py --dataset ${dataset} \
#                    --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                    --output_root data_compressed/${model_size}/${output_version} \
#                    --use_token_scores \
#                    --token_scores_path token_scores/${token_scores_file} \
#                    --target_length ${target_length} \
#                    --use_org_idx \
#                    --question_mode include \
#                    --comp_sent \
#                    --adaptive_sent_comp True \
#                    --sent_comp_ratio 0.1 &

# ### chunk X sentence O tok X sort O               
# python compress.py --dataset ${dataset} \
#                    --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                    --output_root data_compressed/${model_size}/${output_version} \
#                    --use_token_scores \
#                    --token_scores_path token_scores/${token_scores_file} \
#                    --target_length ${target_length} \
#                    --do_sort_ctx \
#                    --use_org_idx \
#                    --question_mode include \
#                    --comp_sent \
#                    --adaptive_sent_comp True \
#                    --sent_comp_ratio 0.1 &
                   

# sent_comp_ratio_list="0 0.05 0.1 0.15 0.2 0.25 0.3"
# sent_comp_ratio_list="0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0"
sent_comp_ratio_list="0.2"

# e_toks="50 100 150 200 250 300"
# e_toks="200"
e_toks="0"
pow_list="1"
# tok_lambs="0.65 0.6 0.55 0.5"
# tok_lambs="1.0 0.7"

## R2C paper report one
# for sent_comp_ratio in ${sent_comp_ratio_list}
# do
#     for e_tok in ${e_toks}
#     do
#         for pow in ${pow_list}
#         do
#             python compress.py --dataset ${dataset} \
#                             --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                             --output_root data_compressed/${model_size}/${output_version} \
#                             --use_token_scores \
#                             --token_scores_path token_scores/${token_scores_file} \
#                             --target_length ${target_length} \
#                             --use_org_idx \
#                             --question_mode include \
#                             --comp_ctx \
#                             --ctx_score_mode mean \
#                             --do_sort_ctx \
#                             --comp_sent \
#                             --sent_comp_ratio ${sent_comp_ratio} \
#                             --adaptive_sent_comp \
#                             --pow ${pow} \
#                             --comp_tok \
#                             --e_tok ${e_tok}
#         done
#     done
# done



for sent_comp_ratio in ${sent_comp_ratio_list}
do
    for e_tok in ${e_toks}
    do
        for pow in ${pow_list}
        do
            python compress.py --dataset ${dataset} \
                            --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
                            --output_root data_compressed/${output_version} \
                            --use_token_scores \
                            --token_scores_path token_scores/${token_scores_file} \
                            --target_length ${target_length} \
                            --use_org_idx \
                            --question_mode include \
                            --comp_ctx \
                            --ctx_score_mode mean \
                            --do_sort_ctx \
                            --comp_sent \
                            --sent_comp_ratio ${sent_comp_ratio} \
                            --adaptive_sent_comp \
                            --pow ${pow} \
                            --comp_tok \
                            --e_tok ${e_tok}
        done
    done
done




####################### OLD

# python compress.py --dataset ${dataset} \
#                 --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                 --output_root data_compressed/${model_size}/${output_version} \
#                 --n_contexts 20 \
#                 --use_token_scores \
#                 --token_scores_path token_scores/${token_scores_file} \
#                 --target_length ${target_length} \
#                 --use_org_idx \
#                 --comp_ctx \
#                 --ctx_score_mode mean \
#                 --question_mode include \
#                 --do_sort_ctx \
#                 --comp_tok \
#                 --tok_lamb ${tok_lamb}


# python compress.py --dataset ${dataset} \
#                 --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                 --output_root data_compressed/${model_size}/${output_version} \
#                 --n_contexts 20 \
#                 --use_token_scores \
#                 --token_scores_path token_scores/${token_scores_file} \
#                 --target_length ${target_length} \
#                 --use_org_idx \
#                 --comp_tok \
#                 --tok_lamb ${tok_lamb}

# tok_lambs="1.0 0.95 0.9 0.85 0.8 0.75 0.7"
# for tok_lamb in ${tok_lambs}
# do
#     python compress.py --dataset ${dataset} \
#                     --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                     --output_root data_compressed/${model_size}/${output_version} \
#                     --n_contexts 20 \
#                     --use_token_scores \
#                     --token_scores_path token_scores/${token_scores_file} \
#                     --target_length ${target_length} \
#                     --comp_ctx \
#                     --ctx_score_mode mean \
#                     --question_mode include \
#                     --pow ${power} \
#                     --do_sort_ctx \
#                     --use_org_idx \
#                     --comp_sent \
#                     --adaptive_sent_comp True \
#                     --sent_comp_ratio ${sent_comp_ratio} \
#                     --comp_tok \
#                     --tok_lamb ${tok_lamb} \
# done


# ctx_score_cumsum=0.45
# power=5
# for tok_lamb in ${tok_lambs}
# do
#     python compress.py --dataset ${dataset} \
#                     --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                     --output_root data_compressed/${model_size}/${output_version} \
#                     --n_contexts 20 \
#                     --use_token_scores \
#                     --token_scores_path token_scores/${token_scores_file} \
#                     --target_length ${target_length} \
#                     --comp_ctx \
#                     --ctx_score_mode mean \
#                     --question_mode include \
#                     --pow ${power} \
#                     --do_sort_ctx \
#                     --use_org_idx \
#                     --comp_sent \
#                     --ctx_score_cumsum ${ctx_score_cumsum} \
#                     --comp_tok \
#                     --tok_lamb ${tok_lamb} \
#                     --adaptive_sent_comp True
# done

# cumsum_list="0.3 0.35 0.4 0.45 0.5 0.55 0.6"
# # cumsum_list="0.45 0.5 0.55 0.6"

# ratio_list="0.05 0.1 0.15 0.2 0.25 0.3"

# dataset="nq_dpr_${split}"
# token_scores_file="${model_size}_token_scores_list_${dataset}_20_ours.pkl"
# # token_scores_file="${model_size}_token_scores_list_${dataset}_20_oneContextFalse.pkl"


# for power in 2
# do

#     for sent_comp_ratio in ${ratio_list}
#     do
#         python compress.py --dataset ${dataset} \
#                         --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                         --output_root data_compressed/${model_size}/${output_version} \
#                         --n_contexts 20 \
#                         --use_token_scores \
#                         --token_scores_path token_scores/${token_scores_file} \
#                         --target_length ${target_length} \
#                         --comp_ctx \
#                         --ctx_score_mode mean \
#                         --question_mode include \
#                         --pow ${power} \
#                         --do_sort_ctx \
#                         --use_org_idx \
#                         --comp_sent \
#                         --sent_comp_ratio ${sent_comp_ratio} \
#                         --adaptive_sent_comp True
#     done

#     for ctx_score_cumsum in ${cumsum_list}
#     do
#         python compress.py --dataset ${dataset} \
#                         --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                         --output_root data_compressed/${model_size}/${output_version} \
#                         --n_contexts 20 \
#                         --use_token_scores \
#                         --token_scores_path token_scores/${token_scores_file} \
#                         --target_length ${target_length} \
#                         --comp_ctx \
#                         --ctx_score_mode mean \
#                         --question_mode include \
#                         --pow ${power} \
#                         --do_sort_ctx \
#                         --use_org_idx \
#                         --comp_sent \
#                         --ctx_score_cumsum ${ctx_score_cumsum} \
#                         --adaptive_sent_comp True
#     done
# done


## For oracle
# python compress.py --dataset ${dataset} \
#                 --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                 --output_root data_compressed/${model_size}/${output_version} \
#                 --n_contexts 20 \
#                 --use_token_scores \
#                 --token_scores_path token_scores/${token_scores_file} \
#                 --target_length ${target_length} \
#                 --use_org_idx \
#                 --comp_sent \
#                 --sent_comp_ratio ${sent_comp_ratio}


# for power in 2 3 4 5 6
# do

#     for sent_comp_ratio in ${ratio_list}
#     do
#         python compress.py --dataset ${dataset} \
#                         --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                         --output_root data_compressed/${model_size}/${output_version} \
#                         --n_contexts 20 \
#                         --use_token_scores \
#                         --token_scores_path token_scores/${token_scores_file} \
#                         --target_length ${target_length} \
#                         --comp_ctx \
#                         --ctx_score_mode mean \
#                         --question_mode include \
#                         --pow ${power} \
#                         --do_sort_ctx \
#                         --use_org_idx \
#                         --comp_sent \
#                         --sent_comp_ratio ${sent_comp_ratio} \
#                         --adaptive_sent_comp True
#     done

#     for ctx_score_cumsum in ${cumsum_list}
#     do
#         python compress.py --dataset ${dataset} \
#                         --input_path ../eun_FiD/open_domain_data/nq/${split}.json \
#                         --output_root data_compressed/${model_size}/${output_version} \
#                         --n_contexts 20 \
#                         --use_token_scores \
#                         --token_scores_path token_scores/${token_scores_file} \
#                         --target_length ${target_length} \
#                         --comp_ctx \
#                         --ctx_score_mode mean \
#                         --question_mode include \
#                         --pow ${power} \
#                         --do_sort_ctx \
#                         --use_org_idx \
#                         --comp_sent \
#                         --ctx_score_cumsum ${ctx_score_cumsum} \
#                         --adaptive_sent_comp True
#     done
# done

# cumsum_list="0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"
# cumsum_list="0.4"

# for ctx_score_cumsum in ${cumsum_list}
# do
#     # dataset="dpr_nq_${split}"
#     # token_scores_file="${model_size}_token_scores_list_${dataset}_${n_contexts}_oneContextFalse.pkl"
#     # python compress.py --dataset ${dataset} \
#     #                 --input_path qa_data/${dataset}.json \
#     #                 --output_root data_compressed/${model_size}/${output_version} \
#     #                 --n_contexts ${n_contexts} \
#     #                 --use_token_scores \
#     #                 --token_scores_path token_scores/${token_scores_file} \
#     #                 --target_length $target_length \
#     #                 --comp_ctx \
#     #                 --ctx_score_mode mean \
#     #                 --question_mode include \
#     #                 --pow ${power}  \
#     #                 --do_sort_ctx \
#     #                 --use_org_idx \
#     #                 --comp_sent \
#     #                 --ctx_score_cumsum ${ctx_score_cumsum} \
#     #                 --adaptive_sent_comp True

#     dataset="dpr_nq_${split}"
#     token_scores_file="${model_size}_token_scores_list_${dataset}_${n_contexts}_oneContextFalse.pkl"
#     python compress.py --dataset ${dataset} \
#                     --input_path qa_data/${dataset}.json \
#                     --output_root data_compressed/${model_size}/${output_version} \
#                     --n_contexts ${n_contexts} \
#                     --use_token_scores \
#                     --token_scores_path token_scores/${token_scores_file} \
#                     --target_length $target_length \
#                     --comp_ctx \
#                     --ctx_score_mode mean \
#                     --question_mode include \
#                     --pow ${power}  \
#                     --do_sort_ctx \
#                     --use_org_idx \
#                     --comp_sent \
#                     --ctx_score_cumsum ${ctx_score_cumsum} \
#                     --adaptive_sent_comp True \
#                     --constraint_1_sent True
# done


