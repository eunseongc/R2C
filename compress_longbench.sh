## compress_longbench bash 예시
# source compress_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_ours 2000 0604_200_t5_s_d_v3_2000_sent0.2_2_ours
# source run_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_ours 1900 0604_200_t5_s_d_v3_1900_sent0.2_2_ours
# source run_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_ours 1800 0604_200_t5_s_d_v3_1800_sent0.2_2_ours
## Longbench pred 예시
# source run.sh base/0604_200_t5_s_d_v3_2000_r2c_chunk_ours && source run.sh base/0604_200_t5_s_d_v3_1900_r2c_chunk_ours && source run.sh base/0604_200_t5_s_d_v3_1800_r2c_chunk_ours && source run.sh base/0604_200_t5_s_d_v3_2000_sent0.2_2_ours && source run.sh base/0604_200_t5_s_d_v3_1900_sent0.2_2_ours && source run.sh base/0604_200_t5_s_d_v3_1800_sent0.2_2_ours


qa_data_version=$1
token_score_version=$2
target_length=$3
output_version=${4:-$token_score_version}

echo "Token_score version: $token_score_version"
echo "Output version: $output_version"

# static
sent_comp_ratio=0.2
power=1
# e_toks="200" ## cst

# datanames="hotpotqa 2wikimqa musique triviaqa trec samsum lcc repobench-p"
datanames="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news triviaqa trec samsum lcc repobench-p"
for dataname in ${datanames}
do
    dataset=longbench_${dataname}

    input_file="${dataset}_test.json"
    # token_scores_file="${token_score_version}/${model_size}_token_scores_list_${dataset}.pkl" ## R2C paper report one
    token_scores_file="${token_score_version}/token_scores_list_${dataset}.pkl"

    ### chunk O sentence O tok X
    python compress.py --dataset ${dataset} \
                        --input_path data/${qa_data_version}/${input_file} \
                        --output_root data_compressed/${model_size}/${output_version} \
                        --use_token_scores \
                        --token_scores_path token_scores/${token_scores_file} \
                        --target_length ${target_length} \
                        --question_mode include \
                        --comp_ctx \
                        --comp_sent \
                        --sent_comp_ratio ${sent_comp_ratio} \
                        --adaptive_sent_comp \
                        --pow ${power} &
done



    # ## chunk X sentence X tok O
    # python compress.py --dataset ${dataset} \
    #                 --input_path data/${qa_data_version}/${input_file} \
    #                 --output_root data_compressed/${model_size}/${output_version} \
    #                 --use_token_scores \
    #                 --token_scores_path token_scores/${token_scores_file} \
    #                 --target_length ${target_length} \
    #                 --question_mode include \
    #                 --comp_tok \
    #                 --tok_lamb 0.95 &

    # ### chunk X sentence O tok X
    # python compress.py --dataset ${dataset} \
    #                 --input_path data/${qa_data_version}/${input_file} \
    #                 --output_root data_compressed/${model_size}/${output_version} \
    #                 --use_token_scores \
    #                 --token_scores_path token_scores/${token_scores_file} \
    #                 --target_length ${target_length} \
    #                 --question_mode include \
    #                 --comp_sent \
    #                 --adaptive_sent_comp True \
    #                 --sent_comp_ratio 0.1 &



# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_cnndaily-base 2000 0821_128_gpt_s_d_a64_cnndaily-base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_ours_base 2000 0821_128_gpt_s_d_a64_ours_base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_ours_large 2000 0821_128_gpt_s_d_a64_ours_large_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_t0-base 2000 0821_128_gpt_s_d_a64_t0-base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_t0-large 2000 0821_128_gpt_s_d_a64_t0-large_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_flan-t5-base 2000 0821_128_gpt_s_d_a64_flan-t5-base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_flan-t5-large 2000 0821_128_gpt_s_d_a64_flan-t5-large_2000