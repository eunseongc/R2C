import os
import json
import tiktoken
import argparse
import numpy as np
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5TokenizerFast
from time import time
from copy import deepcopy
from IPython import embed

_dic={
    "narrativeqa": "Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Now, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "{input}Now, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "Now, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "{input}Now, write a one-page summary of all the news.\n\nSummary:",
    "trec": "{input}",
    "triviaqa": "{input}",
    "samsum": "{input}",
    "passage_count": "{input}Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "The following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "{input}Next line of code:\n",
    "repobench-p": "{input}Next line of code:\n"
}

def split_line(line, tokenizer, max_length, args):

    parts = []
    if args.split_line_uniform:
        if isinstance(tokenizer, T5TokenizerFast):
            len_cur_line = len(tokenizer.encode(line, add_special_tokens=False))
        else:
            len_cur_line = len(tokenizer.encode(line))
        num_splits = (len_cur_line + max_length - 1) // max_length
        if len_cur_line % num_splits == 0:
            target_length = len_cur_line // num_splits
        else:
            target_length = len_cur_line // num_splits + 1        
    else:
        target_length = max_length - 1

    if isinstance(tokenizer, T5TokenizerFast):
        tokens = line.split(' ')
        tokens_org = [t for t in tokens if t != ''] ## Filter out empty strings
        tokens = deepcopy(tokens_org)
        encoded_tokens = tokenizer.encode(line)
        while len(tokens) > 0:
            decoded_part = tokenizer.decode(encoded_tokens[:target_length])
            decoded_part_tokens = decoded_part.split()
            len_decoded_part = len(decoded_part_tokens)
            cur_text = ' '.join(tokens[:len_decoded_part])
            if cur_text.strip() != '':
                parts.append(cur_text)
            ## update_encoded_tokens
            
            tokens = tokens[len_decoded_part:]
            encoded_tokens = tokenizer.encode(' '.join(tokens))

        if ' '.join(parts).split() != ' '.join(tokens_org).split():
            print(">>>>>>>>> WARNING: The split line is not equal to the original line. <<<<<<<<<")
            embed()
    else:
        current_part = []
        tokens = tokenizer.encode(line)

        # parts = [tokenizer.decode(tokens[ii:ii + target_length]) for ii in range(0, len(tokens), target_length)]
        for token in tokens:
            # Check if adding the word exceeds the max length
            if len(current_part) == target_length:
                if current_part:
                    cur_text = tokenizer.decode(current_part).lstrip()
                    parts.append(cur_text)
                    current_part = []
                else:
                    # This handles the case where a single word is longer than max_length WHICH MUST NOT HAPPEN
                    print(">>>>>>>>> WARNING: A single word is longer than the max length. This should not happen. <<<<<<<<<")
                    parts.append(token) ## cur_text == token
                    current_part = []

            current_part.append(token)

        if current_part:
            cur_text = tokenizer.decode(current_part).lstrip()
            if cur_text.strip() == '':
                pass
            else:
                parts.append(cur_text)

    return parts


def process_context(qas, tokenizer, max_textlength, args, title=None, has_answer=None):

    demo2lines = {}
    if args.demoaware:
        if qas['dataset'] in ['narrativeqa', 'qasper', 'multifieldqa_en', 'gov_report', 'qmsum', 'lcc', 'repobench-p']:
            ctx_split_by_line = qas['context'].split('\n')
            demo2lines['d0'] = ctx_split_by_line
        else:
            if qas['dataset'] in ['hotpotqa', '2wikimqa', 'musique', 'multi_news']:
                pattern = r'(Passage \d+:.*?)(?=Passage \d+:|$)'
                
            elif qas['dataset'] == 'trec':
                pattern = r'(Question:.*?Type:.*?)(?=Question:|$)'

            elif qas['dataset'] == 'samsum':
                pattern = r'(Dialogue:.*?Summary:.*?)(?=Dialogue:|$)'
            
            elif qas['dataset'] == 'triviaqa':
                pattern = r'(Question:.*?Answer:.*?Passage:.*?)(?=Question:|$)'

            elif qas['dataset'] in ['passage_count', 'passage_retrieval_en']:
                pattern = r'(Paragraph \d+:.*?)(?=Paragraph|$)'
            
            ctx_split_by_line = re.findall(pattern,  qas['context'], re.DOTALL)
            for i, ctx in enumerate(ctx_split_by_line):
                demo2lines[f"d{i}"] = ctx.split('\n')

    else:
        ctx_split_by_line = qas['context'].split('\n')
        demo2lines['d0'] = ctx_split_by_line

    qas['ctxs'] = []
    for demo_i, ctx_split_by_line in demo2lines.items():
        lin_list = []
        for l_i, lin in enumerate(ctx_split_by_line):
            len_cur_lin = len(tokenizer.encode(lin))
            
            if len_cur_lin > max_textlength:
                # 1. Split long lines if necessary
                ## 1-1. If lin_list is not empty, add the previous lines to the ctxs
                if len(lin_list) > 0:
                    cur_text = '\n'.join(lin_list)
                    if cur_text.strip() == '':
                        pass
                    else:
                        if len(cur_text.split()) > args.add_long_to_previous: ## default: 99999
                            lin = '\n'.join([cur_text, lin])
                        else:
                            ## If cur_text is long enough to be a context, let it be a context
                            ## This way, there will be more contexts than the else case
                            qas['ctxs'].append({"id": f"{demo_i}_{str(l_i-1)}","title": title, "text": cur_text, "has_answer": has_answer})
                    
                ## 1-2. Split the long line, and remain the last part in lin_list
                parts = split_line(lin, tokenizer, max_textlength, args)
                for p_i, part in enumerate(parts):
                    qas['ctxs'].append({"id": f"{demo_i}_{str(l_i)}-{str(p_i)}","title": title, "text": part, "has_answer": has_answer})
                lin_list = []
            else:
                # 2. If the current line is NOT too long,
                lin_list.append(lin)
                cur_text = '\n'.join(lin_list)
                if len(tokenizer.encode(cur_text)) > max_textlength:
                    cur_text = '\n'.join(lin_list[:-1]) ## What if cur_text is ''?
                    if cur_text.strip() == '':
                        cur_text = '\n'.join(lin_list)
                        qas['ctxs'].append({"id": f"{demo_i}_{str(l_i)}","title": title, "text": cur_text, "has_answer": has_answer})
                        lin_list = []
                    else:
                        ## Ordinary case
                        qas['ctxs'].append({"id": f"{demo_i}_{str(l_i-1)}","title": title, "text": cur_text, "has_answer": has_answer})
                        lin_list = [lin_list[-1]]

        cur_text = '\n'.join(lin_list)
        if cur_text.strip() == '':
            pass
        else:
            qas['ctxs'].append({"id": f"{demo_i}_{str(l_i)}","title": title, "text": cur_text, "has_answer": has_answer})
            
    ## Make sure there is not ctx that has an empty string for text
    for ctx in qas['ctxs']:
        if ctx['text'].strip() == '':
            print(">>>>>>>>> WARNING: A context has an empty text. This should not happen. <<<<<<<<<")
            print(qas['context'])
            print(ctx['text'])
            print(ctx['id'])
            print(ctx['title'])
            print(ctx['has_answer'])
            print(">>>>>>>>> WARNING: A context has an empty text. This should not happen. <<<<<<<<<")
            embed()

def main(args):
    max_textlength = args.max_textlength
    split = args.split

    if args.tokenizer == 't5':
        tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    elif args.tokenizer == 'gpt':
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        raise ValueError("Invalid tokenizer")
    
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", \
                "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "multi_news", \
                "trec", "triviaqa", "samsum", \
                "lcc", "repobench-p"]
    
    log = open('log_preprocessing.txt', 'a')
    total_time = 0
    for dataname in datasets:
        if dataname == 'abisee/cnn_dailymail':
            data = load_dataset(dataname, '3.0.0', split=split)
            if split in ['validation', 'test']:
                data = data.select(range(3000))
        else:
            data = load_dataset('THUDM/LongBench', dataname, split=split)
        new_qas = []
        t = time()
        for d_i, d in enumerate(tqdm(data, desc=dataname)):
            qas = {}
            if args.use_dic:
                qas['question'] = _dic[dataname].format(input=d['input'])
            else:
                qas['question'] = d['input']

            qas['context'] = d['context']
            qas['answers'] = d['answers']
            qas['length'] = d['length']
            qas['dataset'] = d['dataset']
            qas['all_classes'] = d['all_classes']
            qas['_id'] = d['_id']

            process_context(qas, tokenizer, max_textlength, args)

            new_qas.append(qas)
        total_time += time() - t
        log.write(f"{dataname}: {time()-t:.3f}s (total: {total_time:.3f}s)\n")
        log.flush()
        output_path = f'data/{args.version}/longbench_{dataname}_{split}.json'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(new_qas, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='temp', help="FiDComp_version", required=False)
    parser.add_argument('--split', type=str, default='test', help="split for dataset", required=False)
    parser.add_argument('--max_textlength', type=int, default=256, required=False)
    parser.add_argument('--tokenizer', type=str, default='t5', required=False)
    parser.add_argument('--use_dic', action='store_true', default=False, required=False)
    parser.add_argument('--split_line_uniform', action='store_true', default=False, required=False)
    parser.add_argument('--add_long_to_previous', type=int, default=999999999, required=False)
    parser.add_argument('--demoaware', action='store_true', default=False, required=False)

    args = parser.parse_args()
    main(args)
    