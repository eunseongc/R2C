gpu=$1
path=$2
split=$3
output_version=$4
## target_length is 500, unless specified
target_length=${5:-500}


CUDA_VISIBLE_DEVICES=${gpu} python eval_fid.py --model_path ${path} \
                                               --eval_data data/nq/${split}.json \
                                               --per_gpu_batch_size 1 \
                                               --write_crossattention_scores \
                                               --write_results \
                                               --text_maxlength 512 \
                                               --n_contexts 20 \
                                               --output_version ${output_version} \
                                               --local-rank -1


dataset="nq_dpr_${split}"
token_score_path="${output_version}/token_scores_list_${dataset}_20.pkl"

sent_comp_ratio_list="0.2"
pow_list="1"

for sent_comp_ratio in ${sent_comp_ratio_list}
do
    for pow in ${pow_list}
    do
        python compress.py --dataset ${dataset} \
                           --input_path data/nq/${split}.json \
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
                           --pow ${pow}
    done
done

