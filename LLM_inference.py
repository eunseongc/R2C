#!/usr/bin/env python3
"""
This code is based on the code from lost-in-the-middle repo ("https://github.com/nelson-liu/lost-in-the-middle")
"""
import sys
import json
import regex
import torch
import string
import random
import logging
import pathlib
import argparse
import statistics
import dataclasses

from copy import deepcopy
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen
from typing import List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)
random.seed(0)

import huggingface_hub
huggingface_hub.login(token="") ## Please fill in your own token


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

class Document:
    title: str
    text: str
    title_rm_answers: Optional[str] = None
    text_rm_answers: Optional[str] = None
    compressed_text: Optional[str] = None
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls, data):
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def get_qa_prompt(
    question: str, documents: List[Document],
):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")        
        
    return QA_PROMPT.format(question=question, search_results="\n".join(formatted_documents))


def get_closedbook_qa_prompt(question: str):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")

    return CLOSED_QA_PROMPT.format(question=question)









############################################################################
METRICS = [
    (best_subspan_em, "best_subspan_em"),
]
T = TypeVar("T")

QA_PROMPT="""Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

{search_results}

Question: {question}
Answer:"""

CLOSED_QA_PROMPT="""Question: {question}
Answer:"""
############################################################################


def main(
    input_path,
    compressed,
    model_name,
    temperature,
    top_p,
    closedbook,
    num_gpus,
    hf_cache_path,
    output_path,
    max_new_tokens,
    max_prompt_length,
    n_psgs: int,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for i, line in enumerate(tqdm(fin)):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            documents = []
            if compressed:
                prompt = input_example["compressed_prompt"]
            else:
                if closedbook:
                    prompt = get_closedbook_qa_prompt(
                        question
                    )
                else:
                    if n_psgs is not None:
                        input_example["ctxs"] = input_example["ctxs"][:n_psgs]
                        
                    for ctx in deepcopy(input_example["ctxs"]):
                        documents.append(Document.from_dict(ctx))

                    if not documents:
                        raise ValueError(f"Did not find any documents for example: {input_example}")

                    prompt = get_qa_prompt(
                        question,
                        documents,
                    )

            if "chat" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
                prompt = format_chat_prompt(prompt)

            prompt_length = len(tokenizer(prompt)["input_ids"])
            if max_prompt_length < prompt_length:
                logger.info(
                    f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
                    f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)
    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    logger.info("Loading model")
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        download_dir=hf_cache_path,
        load_format="pt",
        max_num_batched_tokens=max_prompt_length,
    )
    logger.info("Model loaded")
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    raw_responses = model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in raw_responses]

    all_example_metrics = []
    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            f.write(json.dumps(output_example) + "\n")
            all_example_metrics.append(get_metrics_for_example(output_example))
    
    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] * 100 for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name}: {average_metric_value}")


def get_metrics_for_example(example):
    if example.get("answers") is None:
        example["answers"] = example["answer"]
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


def format_chat_prompt(message: str):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
    )
    lines = ["<s>[INST] <<SYS>>", DEFAULT_SYSTEM_PROMPT, "<</SYS>>", "", f"{message} [/INST]"]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--compressed", help="Whether the input is compressed", action="store_true", default=False)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
        choices=[
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
        ],
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top_p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )

    parser.add_argument("--num_gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--hf_cache_path", help="Path to huggingface cache to use.")
    parser.add_argument("--output_path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max_new_tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_prompt_length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=4096,
    )
    parser.add_argument("--n_psgs", help="n_psgs", type=int, default=None)
    
    
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.compressed,
        args.model,
        args.temperature,
        args.top_p,
        args.closedbook,
        args.num_gpus,
        args.hf_cache_path,
        args.output_path,
        args.max_new_tokens,
        args.max_prompt_length,
        args.n_psgs,
    )
    logger.info("finished running %s", sys.argv[0])
