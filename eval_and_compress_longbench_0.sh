# source eval_and_compress_longbench_0.sh 0 checkpoints_fid/nq_cnn_v2_20_pair_t5-base_0905w_largesteps_0/checkpoint/step-896000/ 0609_128_gpt_s_d_a64 0609_128_gpt_s_d_a64_nq_cnn_v2_0905w_largesteps_0_step896000

gpu=$1
path=$2
input_version=$3
output_version=$4
## target_length is 2000, unless specified
target_length=${5:-2000}

# datanames="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report"
# datanames="qmsum multi_news triviaqa trec samsum lcc repobench-p"
datanames="gov_report"

echo "datanames: ${datanames}"
for dataname in $datanames
do
    dataset=longbench_${dataname}
    CUDA_VISIBLE_DEVICES=${gpu} python eval_fid.py --model_path ${path} \
                                                   --eval_data data/${input_version}/${dataset}_test.json \
                                                   --per_gpu_batch_size 1 \
                                                   --write_crossattention_scores \
                                                   --n_contexts 9999 \
                                                   --text_maxlength 512 \
                                                   --output_version ${output_version} \
                                                   --local-rank -1
done


sent_comp_ratio=0.2
power=1
for dataname in ${datanames}
do
    dataset=longbench_${dataname}

    # token_scores_file="${token_score_version}/${model_size}_token_scores_list_${dataset}.pkl" ## R2C paper report one

    ### chunk O sentence O tok X
    python compress.py --dataset ${dataset} \
                       --input_path data/${input_version}/${dataset}_test.json \
                       --output_root data_compressed/${model_size}/${output_version} \
                       --use_token_scores \
                       --token_scores_path token_scores/${output_version}/token_scores_list_${dataset}.pkl \
                       --target_length ${target_length} \
                       --question_mode include \
                       --comp_ctx \
                       --comp_sent \
                       --sent_comp_ratio ${sent_comp_ratio} \
                       --adaptive_sent_comp \
                       --pow ${power} &
done