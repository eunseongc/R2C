

data_dir="data_compressed"
pred_dir="pred_compressed"

# model_name="Llama-2-7b-chat-hf"
model_name="Meta-Llama-3-8B-Instruct"
# meta-llama/
if [ "${model_name}" == "llama-30b-instruct" ]; then
    dist="upstage"
elif [ "${model_name}" == "longchat-13b-16k" ]; then
    dist="lmsys"
elif [ "${model_name}" == "Llama-2-13b-chat-hf" ]; then
    dist="meta-llama"
elif [ "${model_name}" == "Llama-2-7b-chat-hf" ]; then
    dist="meta-llama"
elif [ "${model_name}" == "Meta-Llama-3-8B-Instruct" ]; then
    dist="meta-llama"
else
    echo "Invalid model name"
    return 1
fi

# data_path="nq_cnn_v2_0904w_0_step320000/nq_dpr_dev_20/fid_500_ctxTrue_sortTrue_sentTrue0.2_1_tokFalse0.jsonl.gz"
# input_path="${data_dir}/${data_path}"
# output_path="${pred_dir}/${data_path}_${model_name}"

# echo ">>>>>>>>>>>>>>>>>> Start predict: ${output_path}"
# python -u ./scripts/get_qa_responses_from_llama_2.py \
#     --input-path ${input_path} \
#     --is-compressed True \
#     --max-new-tokens 100 \
#     --num-gpus 2 \
#     --model ${dist}/${model_name} \
#     --output-path ${output_path}
# echo ">>>>>>>>>>>>>>>>>> Just finished: ${output_path}"

# data_path="nq_cnn_v2_0905w_largesteps_0_step656000/nq_dpr_dev_20/fid_500_ctxTrue_sortTrue_sentTrue0.2_1_tokFalse0.jsonl.gz"
# input_path="${data_dir}/${data_path}"
# output_path="${pred_dir}/${data_path}_${model_name}"

# echo ">>>>>>>>>>>>>>>>>> Start predict: ${output_path}"
# python -u ./scripts/get_qa_responses_from_llama_2.py \
#     --input-path ${input_path} \
#     --is-compressed True \
#     --max-new-tokens 100 \
#     --num-gpus 2 \
#     --model ${dist}/${model_name} \
#     --output-path ${output_path}
# echo ">>>>>>>>>>>>>>>>>> Just finished: ${output_path}"

# data_path="500_ours-base/nq_dpr_test3610_20/test_oracle.jsonl.gz"
# data_path="500_ours-base/nq_dpr_test3610_20/test_oracle_llama3_need_ctx_llama3attentions.json"
# # data_path="icae/llama3_need_ctx_testoracle/icae_120_ctxFalse_sortFalse_sentTrue0.2_1_tokFalse0.jsonl.gz"
# # data_path="500_ours-base/llama3_need_ctx_testoracle_20/fid_120_ctxFalse_sortFalse_sentTrue0.2_1_tokFalse0.jsonl.gz"
# input_path="${data_dir}/${data_path}"
# output_path="${pred_dir}/${data_path}_${model_name}"

# echo ">>>>>>>>>>>>>>>>>> Start predict: ${output_path}"
# # CUDA_VISIBLE_DEVICES=0 python -u ./scripts/get_qa_responses_from_llama_2.py \
# python -u ./scripts/get_qa_responses_eun.py \
#     --input-path ${input_path} \
#     --is-compressed True \
#     --max-new-tokens 100 \
#     --num-gpus 2 \
#     --model ${dist}/${model_name} \
#     --output-path ${output_path}
# echo ">>>>>>>>>>>>>>>>>> Just finished: ${output_path}"

# data_path="icae/llama3_need_ctx_testoracle/icae_70_ctxFalse_sortFalse_sentTrue0.2_1_tokFalse0.jsonl.gz"
# data_path="500_ours-base/llama3_need_ctx_testoracle_20/fid_70_ctxFalse_sortFalse_sentTrue0.2_1_tokFalse0.jsonl.gz"
# input_path="${data_dir}/${data_path}"
# output_path="${pred_dir}/${data_path}_${model_name}"


# echo ">>>>>>>>>>>>>>>>>> Start predict: ${output_path}"
# # CUDA_VISIBLE_DEVICES=0 python -u ./scripts/get_qa_responses_from_llama_2.py \
# python -u ./scripts/get_qa_responses_eun.py \
#     --input-path ${input_path} \
#     --is-compressed True \
#     --max-new-tokens 100 \
#     --num-gpus 2 \
#     --model ${dist}/${model_name} \
#     --output-path ${output_path}
# echo ">>>>>>>>>>>>>>>>>> Just finished: ${output_path}"


######################## ORIGINAL PROMPT ########################
data_path="500_ours-base/nq_dpr_test3610_20/test_oracle_llama3_need_ctx_llama3attentions.json"
# data_path="500_ours-base/nq_dpr_test3610_20/fid_500_ctxTrue_sortTrue_sentTrue0.2_1_tokFalse0.jsonl.gz"
input_path="${data_dir}/${data_path}"

for token_limit in 30 40 50 60 70 80 90 100 110 120 130 140 150; do
    output_path="${pred_dir}/${data_path}_${model_name}_${token_limit}tokens"
    echo ">>>>>>>>>>>>>>>>>> Start predict: ${output_path}"
    python -u ./scripts/get_qa_responses_eun.py \
        --input-path ${input_path} \
        --is-compressed True \
        --max-new-tokens 100 \
        --num-gpus 2 \
        --model ${dist}/${model_name} \
        --token_limit ${token_limit} \
        --output-path ${output_path}
    echo ">>>>>>>>>>>>>>>>>> Just finished: ${output_path}"
done