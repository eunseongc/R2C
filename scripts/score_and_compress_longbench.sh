gpu=$1
path=$2
version=$3
## target_length is 2000, unless specified
target_length=${4:-2000}
datanames="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p"
    
echo "datanames: ${datanames}"
for dataname in $datanames
do
    dataset="longbench_${dataname}"
    token_scores_path="token_scores/${version}/token_scores_list_${dataset}.pkl"
    CUDA_VISIBLE_DEVICES=${gpu} python eval_fid.py --model_path ${path} \
                                                   --eval_data data/${version}/${dataset}_test.json \
                                                   --per_gpu_batch_size 1 \
                                                   --write_crossattention_scores \
                                                   --token_scores_path ${token_scores_path} \
                                                   --text_maxlength 512 \
                                                   --local-rank -1
done


sent_comp_ratio=0.2
pow=1
for dataname in ${datanames}
do
    dataset="longbench_${dataname}"
    python compress.py --dataset ${dataset} \
                       --input_path data/${version}/${dataset}_test.json \
                       --output_root data_compressed/${version} \
                       --use_token_scores \
                       --token_scores_path token_scores/${version}/token_scores_list_${dataset}.pkl \
                       --target_length ${target_length} \
                       --question_mode include \
                       --comp_ctx \
                       --ctx_score_mode mean \
                       --comp_sent \
                       --sent_comp_ratio ${sent_comp_ratio} \
                       --adaptive_sent_comp \
                       --pow ${pow} &
done