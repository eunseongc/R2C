## compress_longbench bash 예시
# source compress_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_ours 2000 0604_200_t5_s_d_v3_2000_sent0.2_2_ours && source run_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_ours 1900 0604_200_t5_s_d_v3_1900_sent0.2_2_ours && source run_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_ours 1800 0604_200_t5_s_d_v3_1800_sent0.2_2_ours
## Longbench pred 예시
# source run.sh base/0604_200_t5_s_d_v3_2000_r2c_chunk_ours && source run.sh base/0604_200_t5_s_d_v3_1900_r2c_chunk_ours && source run.sh base/0604_200_t5_s_d_v3_1800_r2c_chunk_ours && source run.sh base/0604_200_t5_s_d_v3_2000_sent0.2_2_ours && source run.sh base/0604_200_t5_s_d_v3_1900_sent0.2_2_ours && source run.sh base/0604_200_t5_s_d_v3_1800_sent0.2_2_ours


qa_data_version=$1
token_score_version=$2
target_length=$3
output_version=${4:-$token_score_version}

# token_score_version=0503_100
# output_version=0503_100

echo "Token_score version: $token_score_version"
echo "Output version: $output_version"


# list1="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"
list1="gov_report"


# list1="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc"

# model_size=large

# static
sent_comp_ratio=0.2
power=1
# e_toks="200" ## cst

# Llama-2-7b-chat-hf_fid_500_ctxTrue_sortTrue_sentTrue0.1_2_tokTrue300.jsonl.gz.jsonl.g
# dynamic
# ctx_score_cumsum=0.45
# power=5

# tok_lamb_list="1.0 0.7"
# tok_lamb_list="0.7" ## cst
# tok_lamb_list="1.0" ## cs



for dataname in ${list1}
do
    dataset=longbench_${dataname}

    input_file="${dataset}_test.json"
    ## R2C paper report one
    # token_scores_file="${token_score_version}/${model_size}_token_scores_list_${dataset}.pkl"
    token_scores_file="${token_score_version}/token_scores_list_${dataset}.pkl"


    # ## chunk X sentence X tok O
    # python compress.py --dataset ${dataset} \
    #                 --input_path ../eun_FiD/open_domain_data/${qa_data_version}/${input_file} \
    #                 --output_root data_compressed/${model_size}/${output_version} \
    #                 --use_token_scores \
    #                 --token_scores_path token_scores/${token_scores_file} \
    #                 --target_length ${target_length} \
    #                 --question_mode include \
    #                 --comp_tok \
    #                 --tok_lamb 0.95 &

    # ### chunk X sentence O tok X
    # python compress.py --dataset ${dataset} \
    #                 --input_path ../eun_FiD/open_domain_data/${qa_data_version}/${input_file} \
    #                 --output_root data_compressed/${model_size}/${output_version} \
    #                 --use_token_scores \
    #                 --token_scores_path token_scores/${token_scores_file} \
    #                 --target_length ${target_length} \
    #                 --question_mode include \
    #                 --comp_sent \
    #                 --adaptive_sent_comp True \
    #                 --sent_comp_ratio 0.1 &


    ### chunk O sentence O tok X
    python compress.py --dataset ${dataset} \
                        --input_path ../eun_FiD/open_domain_data/${qa_data_version}/${input_file} \
                        --output_root data_compressed/${model_size}/${output_version} \
                        --use_token_scores \
                        --token_scores_path token_scores/${token_scores_file} \
                        --target_length ${target_length} \
                        --question_mode include \
                        --comp_ctx \
                        --comp_sent \
                        --sent_comp_ratio ${sent_comp_ratio} \
                        --adaptive_sent_comp \
                        --pow ${power}


    # python compress.py --dataset ${dataset} \
    #                 --input_path ../eun_FiD/open_domain_data/${qa_data_version}/${input_file} \
    #                 --output_root data_compressed/${model_size}/${output_version} \
    #                 --use_token_scores \
    #                 --token_scores_path token_scores/${token_scores_file} \
    #                 --target_length ${target_length} \
    #                 --question_mode include \
    #                 --comp_tok \
    #                 --e_tok 100
done



# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_cnndaily-base 2000 0821_128_gpt_s_d_a64_cnndaily-base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_ours_base 2000 0821_128_gpt_s_d_a64_ours_base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_ours_large 2000 0821_128_gpt_s_d_a64_ours_large_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_t0-base 2000 0821_128_gpt_s_d_a64_t0-base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_t0-large 2000 0821_128_gpt_s_d_a64_t0-large_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_flan-t5-base 2000 0821_128_gpt_s_d_a64_flan-t5-base_2000 &
# source run_longbench.sh 0609_128_gpt_s_d_a64 0821_128_gpt_s_d_a64_flan-t5-large 2000 0821_128_gpt_s_d_a64_flan-t5-large_2000