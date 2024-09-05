# source run_test_1_longbench.sh checkpoints/nq_cnn_20_pair_t5-base_0830_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a64 0609_128_gpt_s_d_a64_nq_cnn_0830
# source run_test_1_longbench.sh checkpoints/nq_cnn_20_pair_t5-base_position_0830_0/checkpoint/step-288000/ 0609_128_gpt_s_d_a64 0609_128_gpt_s_d_a64_nq_cnn_posiion_0830

gpu=$1
path=$2
input_version=$3
output_version=$4

## singledocqa: narrativeqa qasper multifieldqa_en
## multidocqa: hotpotqa 2wikimqa musique
## summarization: gov_report qmsum multi_news
## fewshot: triviaqa trec samsum
## code: lcc repobench-p

# if dataname in ["narrativeqa", "qasper", "multifieldqa_en"]:
#     task="singledocqa"
# elif dataname in ["hotpotqa", "2wikimqa", "musique"]:
#     task="multidocqa"
# elif dataname in ["gov_report", "qmsum", "multi_news"]:
#     task="summarization"
# elif dataname in ["triviaqa", "trec", "samsum"]:
#     task="fewshot"
# elif dataname in ["lcc", "repobench-p"]:
#     task="code"
# narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news triviaqa trec samsum lcc repobench-p


# datanames="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report"
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


# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_100_gpt_s_d 0609_100_gpt_s_d_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_100_gpt_s_d_a25 0609_100_gpt_s_d_a25_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_100_gpt_s_d_a50 0609_100_gpt_s_d_a50_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d 0609_128_gpt_s_d_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a25 0609_128_gpt_s_d_a25_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a50 0609_128_gpt_s_d_a50_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a64 0609_128_gpt_s_d_a64_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d 0609_150_gpt_s_d_ours && ################### 이거 다시해야하듯
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d_a25 0609_150_gpt_s_d_a25_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d_a75 0609_150_gpt_s_d_a75_ours &&
# source run_test_0_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d_a100 0609_150_gpt_s_d_a100_ours

# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_100_gpt_s_d 0609_100_gpt_s_d_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_100_gpt_s_d_a25 0609_100_gpt_s_d_a25_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_100_gpt_s_d_a50 0609_100_gpt_s_d_a50_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d 0609_128_gpt_s_d_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a25 0609_128_gpt_s_d_a25_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a50 0609_128_gpt_s_d_a50_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_128_gpt_s_d_a64 0609_128_gpt_s_d_a64_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d 0609_150_gpt_s_d_ours && ################### 이거 다시해야하듯
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d_a25 0609_150_gpt_s_d_a25_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d_a75 0609_150_gpt_s_d_a75_ours &&
# source run_test_1_longbench.sh checkpoints/nq_20_0/checkpoint/best_dev/ 0609_150_gpt_s_d_a100 0609_150_gpt_s_d_a100_ours

## 0609_150_gpt_s_d/ == 0609_150_gpt_s_d_a50임

# 0609_100_gpt_s_d 0609_100_gpt_s_d_ours
# 0609_100_gpt_s_d_a25 0609_100_gpt_s_d_a25_ours
# 0609_100_gpt_s_d_a50 0609_100_gpt_s_d_a50_ours
# 0609_128_gpt_s_d 0609_128_gpt_s_d_ours
# 0609_128_gpt_s_d_a25 0609_128_gpt_s_d_a25_ours
# 0609_128_gpt_s_d_a50 0609_128_gpt_s_d_a50_ours
# 0609_128_gpt_s_d_a64 0609_128_gpt_s_d_a64_ours
# 0609_150_gpt_s_d_a 0609_150_gpt_s_d_a_ours
# 0609_150_gpt_s_d_a25 0609_150_gpt_s_d_a25_ours
# 0609_150_gpt_s_d_a75 0609_150_gpt_s_d_a75_ours
# 0609_150_gpt_s_d_a100 0609_150_gpt_s_d_a100_ours

