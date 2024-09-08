

data_dir="data_compressed"
pred_dir="pred_compressed"

model_name="Llama-2-7b-chat-hf"
if [ "${model_name}" == "llama-30b-instruct" ]; then
    dist="upstage"
elif [ "${model_name}" == "longchat-13b-16k" ]; then
    dist="lmsys"
elif [ "${model_name}" == "Llama-2-13b-chat-hf" ]; then
    dist="meta-llama"
elif [ "${model_name}" == "Llama-2-7b-chat-hf" ]; then
    dist="meta-llama"
else
    echo "Invalid model name"
    return 1
fi

input_path="${data_dir}/nq_cnn_v2_0904w_0_step320000/nq_dpr_dev_20/fid_500_ctxTrue_sortTrue_sentTrue0.2_1_tokFalse0.jsonl.gz"
output_path="${pred_dir}/{input_path}_${model_name}"

echo ">>>>>>>>>>>>>>>>>> Start predict: ${output_path}"
python -u ./scripts/get_qa_responses_from_llama_2.py \
    --input-path ${input_path} \
    --is-compressed True \
    --max-new-tokens 100 \
    --num-gpus 2 \
    --model ${dist}/${model_name} \
    --output-path ${output_path}
echo ">>>>>>>>>>>>>>>>>> Just finished: ${output_path}"