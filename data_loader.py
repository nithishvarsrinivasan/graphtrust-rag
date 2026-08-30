"""
data_loader.py
---------------
Extracted from your notebook cells 11-13. Logic unchanged.
"""

import json
import os
import random

import config


def load_hotpotqa_subset():
    from datasets import load_dataset

    random.seed(config.RANDOM_SEED)
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT)

    corpus = []
    questions = []
    passage_id = 0

    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(questions) >= config.NUM_TEST_QUESTIONS and len(corpus) >= config.NUM_CORPUS_PASSAGES:
            break

        example = ds[idx]
        titles = example["context"]["title"]
        sentence_lists = example["context"]["sentences"]

        for title, sentences in zip(titles, sentence_lists):
            if len(corpus) >= config.NUM_CORPUS_PASSAGES:
                break
            paragraph_text = f"{title}. " + " ".join(sentences)
            corpus.append({"id": passage_id, "text": paragraph_text})
            passage_id += 1

        if len(questions) < config.NUM_TEST_QUESTIONS:
            questions.append({
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
            })

    return corpus, questions


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare_corpus_and_questions(force_rebuild=False):
    if (not force_rebuild
            and os.path.exists(config.CORPUS_JSONL_PATH)
            and os.path.exists(config.QUESTIONS_JSONL_PATH)):
        corpus = load_jsonl(config.CORPUS_JSONL_PATH)
        questions = load_jsonl(config.QUESTIONS_JSONL_PATH)
        print(f"Loaded cached corpus ({len(corpus)} passages) and "
              f"questions ({len(questions)}) from disk.")
        return corpus, questions

    print("Building corpus and questions from HotpotQA...")
    corpus, questions = load_hotpotqa_subset()
    save_jsonl(corpus, config.CORPUS_JSONL_PATH)
    save_jsonl(questions, config.QUESTIONS_JSONL_PATH)
    print(f"Saved {len(corpus)} passages and {len(questions)} questions.")
    return corpus, questions


if __name__ == "__main__":
    corpus, questions = prepare_corpus_and_questions()
    print("\nSample passage:", corpus[0])
    print("\nSample question:", questions[0])
