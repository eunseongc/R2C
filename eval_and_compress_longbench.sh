datanames="qmsum multi_news triviaqa trec samsum lcc repobench-p"

echo "datanames: ${datanames}"
for dataname in $datanames
do
    CUDA_VISIBLE_DEVICES=${gpu} python inference_fid.py --model_path ${path} \
                                                        --eval_data data/${input_version}/longbench_${dataname}_test.json \
                                                        --per_gpu_batch_size 1 \
                                                        --write_crossattention_scores \
                                                        --n_contexts 9999 \
                                                        --text_maxlength 512 \
                                                        --output_version ${output_version} \
                                                        --local-rank -1
done
