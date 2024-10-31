tokenizer=gpt
max_textlength=100
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a25 --add_long_to_previous 25 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a50 --add_long_to_previous 50 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &

max_textlength=128
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a25 --add_long_to_previous 25 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a50 --add_long_to_previous 50 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
# Reported one
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a64 --add_long_to_previous 64 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &

max_textlength=150
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a25 --add_long_to_previous 25 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a75 --add_long_to_previous 75 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware &
python convert_longbench_to_fid.py --version 0609_${max_textlength}_${tokenizer}_s_d_a100 --add_long_to_previous 100 --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware


## v2: demo aware 추가
## v3: line split in demo 

## [아래 실험 후 결론] T5, 150 토큰이 옳음.
# for max_textlength in 150 200 250
# do
#     for tokenizer in gpt t5
#     do
#         python convert_longbench_to_fid.py --version 0531_${max_textlength}_${tokenizer} --tokenizer ${tokenizer} --max_textlength ${max_textlength} &
#         python convert_longbench_to_fid.py --version 0531_${max_textlength}_${tokenizer}_s_a --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --add_long_to_previous &
#         python convert_longbench_to_fid.py --version 0531_${max_textlength}_${tokenizer}_s --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform &
#         python convert_longbench_to_fid.py --version 0531_${max_textlength}_${tokenizer}_a --tokenizer ${tokenizer} --max_textlength ${max_textlength} --add_long_to_previous &
#     done
# done



# source run_test_0_longbench.sh checkpoints/nq_reader_base/ 0602_150_t5_s_d_v2 0602_150_t5_s_d_v2
# source run_test_1_longbench.sh checkpoints/nq_reader_base/ 0602_200_t5_s_d_v2 0602_200_t5_s_d_v2


# source run_longbench.sh 0603_150_t5_s_d_v2 0603_150_t5_s_d_v2 && source run_longbench.sh 0602_150_t5_s_a_d 0602_150_t5_s_a_d && source run_longbench.sh 0602_200_t5_s_d 0602_200_t5_s_d && source run_longbench.sh 0602_200_t5_s_a_d 0602_200_t5_s_a_d
# sleep 40m && source run.sh base/0603_100_t5_s_d_v2_sent0.1_2 && source run.sh base/0603_128_t5_s_d_v2_sent0.1_2 && source run.sh base/0603_150_t5_s_d_v2_sent0.1_2 && source run.sh base/0603_200_t5_s_d_v2_sent0.1_2

# source run_longbench.sh 0603_128_t5_s_d_v3 0603_128_t5_s_d_v3 0603_128_t5_s_d_v3_sent0.1_2 && source run_longbench.sh 0603_150_t5_s_d_v3 0603_150_t5_s_d_v3 0603_150_t5_s_d_v3_sent0.1_2 && source run_longbench.sh 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3 0603_200_t5_s_d_v3_sent0.1_2


# source run.sh base/0603_100_t5_s_d_v2_sent0.1_2 && source run.sh base/0603_1200_t5_s_d_v2_sent0.1_2 && source run.sh base/0603_150_t5_s_d_v2_sent0.1_2 && source run.sh base/0603_200_t5_s_d_v2_sent0.1_2
