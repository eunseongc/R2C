dataset="nq"
n_contexts=20 ## Number of contexts to use for each question
gpu=$1
fid_path=$2
eval_data=$3 # e.g., data/nq/${split}.json
token_scores_path=$4 # path to save the token scores using eval_fid.py


CUDA_VISIBLE_DEVICES=${gpu} python eval_fid.py --model_path ${fid_path} \
                                               --eval_data ${eval_data} \
                                               --per_gpu_batch_size 1 \
                                               --write_crossattention_scores \
                                               --token_scores_path ${token_scores_path} \
                                               --text_maxlength 512 \
                                               --n_contexts ${n_contexts} \
                                               --local-rank -1



target_length=${5:-500} # target_length is 500, unless specified

sent_comp_ratio=0.2
pow=1
python compress.py --dataset ${dataset} \
                   --input_path ${eval_data} \
                   --n_contexts ${n_contexts} \
                   --output_root data_compressed/nq \
                   --use_token_scores \
                   --token_scores_path ${token_scores_path} \
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