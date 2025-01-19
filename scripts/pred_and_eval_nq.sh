MODEL_NAME=Llama-2-7b-chat-hf
MODEL=meta-llama/${MODEL_NAME}
INPUT_PATH=data_compressed/nq/nq_20/fid_500_ctxTrue_sortTrue_sentTrue0.2_1_tokFalse0.jsonl.gz

CUDA_VISIBLE_DEVICES=0 python LLM_inference.py \
    --input_path ${INPUT_PATH} \
    --compressed \
    --model ${MODEL} \
    --num_gpus 1 \
    --output_path pred/nq/${INPUT_PATH}_${MODEL_NAME}_prediction.jsonl.gz \
    --max_new_tokens 100