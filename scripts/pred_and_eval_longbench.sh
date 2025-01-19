VERSION=$1 # 128_gpt_s_d_a64
MODEL=llama2-7b-chat

CUDA_VISIBLE_DEVICES=1 python src_longbench/pred.py --version ${VERSION} \
                             --model ${MODEL}

python src_longbench/eval.py --input_dir pred/${MODEL}/${VERSION}