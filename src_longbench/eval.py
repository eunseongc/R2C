import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

dataset2numsample = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "passage_retrieval_en": 200,
    "passage_count": 200,
    "lcc": 500,
    "repobench-p": 500,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None, help="input directory to evaluate", required=False)
    return parser.parse_args(args)


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()

    if args.input_dir is None:
        raise ValueError("Please specify the input directory to evaluate")
    
    ## all_files except for result.json
    all_files = [f for f in os.listdir(args.input_dir) if f != "result.json"]
    
    dataset_ordered = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p", "result"]
    ## Order the dataset in the order of the leaderboard
    ## filename includes the dataset name
    all_files = sorted(all_files, key=lambda x: dataset_ordered.index(x.split('.')[0]))

    out_path = os.path.join(args.input_dir, "result.json")    
    if os.path.exists(out_path):
        print(f">> Loading existing scores, {out_path}")
        scores = json.load(open(out_path, "r"))
        datasets_already_evaluated = set(list(scores.keys()))
        print(f">> Skipping datasets already evaluated: {datasets_already_evaluated}")
        all_files = [f for f in all_files if f.split('.')[0] not in datasets_already_evaluated]


    print("Evaluating on:", all_files)
    for filename in tqdm(all_files):
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        
        num_samples = 0
        with open(os.path.join(args.input_dir, filename), "r", encoding="utf-8") as f:
            for lin in f:
                num_samples += 1
        if num_samples != dataset2numsample[dataset]:
            if dataset == 'repobench-p' and num_samples == 496:
                pass
            elif dataset == 'multifieldqa_en' and num_samples == 147:
                pass
            else:
                print(f"Warning: {dataset} has {num_samples} samples, expected {dataset2numsample[dataset]}")
                continue

        with open(os.path.join(args.input_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                # all_classes = ["Food", "Date", "Order, rank", "Speed", "Disease and medicine", "Word with a special property", "Abbreviation", "Language", "Letter like a-z", "Other entity", "Animal", "Expression abbreviated", "Price", "Techniques and method", "Musical instrument", "Mountain", "Currency name", "Event", "Product", "State", "Individual", "Organ of body", "Reason", "Manner of an action", "City", "Religion", "Invention, book and other creative piece", "Distance, linear measure", "Temperature", "Postcode or other code", "Size, area and volume", "Sport", "Country", "Other location", "Lasting time of somethin", "Equivalent term", "Description of something", "Weight", "Vehicle", "Color", "Other number", "Definition of something", "Element and substance", "Description of a person", "Symbols and sign", "Number of something", "Plant", "Percent, fraction", "Group or organization of person", "Title of a person"]
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])

        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

    ## Reorder the scores in the order of the dataset_ordered;
    scores = {k: scores[k] for k in dataset_ordered if scores.get(k) is not None}
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
