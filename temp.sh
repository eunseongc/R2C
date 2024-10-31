path=$1
split=$2

export NGPU=2
for n_context in 1 5 10 20 50 100
do
    python -m torch.distributed.launch --nproc_per_node=${NGPU} eval_fid.py --model_path ${path} \
                       --eval_data data/nq/${split}.json \
                       --per_gpu_batch_size 1 \
                       --text_maxlength 250 \
                       --mode pair \
                       --n_contexts ${n_context}
done