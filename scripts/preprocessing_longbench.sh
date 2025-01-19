max_textlength=$1
tokenizer=$2
add_long_to_previous=$3
python convert_longbench_to_fid.py --version ${max_textlength}_${tokenizer}_s_d_a${add_long_to_previous} --add_long_to_previous ${add_long_to_previous} --tokenizer ${tokenizer} --max_textlength ${max_textlength} --split_line_uniform --demoaware