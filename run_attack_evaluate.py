#pull from git and run in colab
import json
import shutil

import config
from pipeline import BaseRAGPipeline
from attack import inject_and_evaluate


def main():
    with open("attack_output.json") as f:
        attack_data = json.load(f)

    question = attack_data["question"]
    true_answer = attack_data["true_answer"]
    target_false_answer = attack_data["target_false_answer"]
    adversarial_passage = attack_data["adversarial_passage"]

    print("=" * 70)
    print("STEP 1: BASELINE (before poisoning)")
    print("=" * 70)
    pipeline = BaseRAGPipeline()
    baseline_result = pipeline.ask(question)
    baseline_answer = baseline_result["answer"]

    print(f"\nTrue answer (dataset): {true_answer}")
    print(f"Baseline pipeline answer: {baseline_answer}")

    print("\n" + "=" * 70)
    print("STEP 2: INJECT ADVERSARIAL PASSAGE + RE-INDEX")
    print("=" * 70)
    print(f"Adversarial passage:\n{adversarial_passage}\n")

    # Force a clean re-index - required, since ColBERT won't just add one
    # new passage to an existing index.
    shutil.rmtree(config.COLBERT_INDEX_ROOT, ignore_errors=True)

    evaluation = inject_and_evaluate(
        pipeline=pipeline,
        question=question,
        target_false_answer=target_false_answer,
        adversarial_passage=adversarial_passage,
    )

    print("\n" + "=" * 70)
    print("STEP 3: BEFORE / AFTER COMPARISON")
    print("=" * 70)
    print(f"Question:              {question}")
    print(f"True answer:           {true_answer}")
    print(f"Baseline answer:       {baseline_answer}")
    print(f"Target false answer:   {target_false_answer}")
    print(f"Post-attack answer:    {evaluation['final_answer']}")
    print(f"Adversarial passage retrieved into top-k: {evaluation['was_retrieved']}")
    print(f"Attack succeeded (false answer in output): {evaluation['attack_succeeded']}")

    answer_changed = baseline_answer.strip().lower() != evaluation["final_answer"].strip().lower()
    print(f"\nDid the answer change at all: {answer_changed}")

    with open("attack_evaluation_result.json", "w") as f:
        json.dump({
            **evaluation,
            "baseline_answer": baseline_answer,
            "true_answer": true_answer,
            "answer_changed": answer_changed,
        }, f, indent=2)

    print("\nSaved full result to attack_evaluation_result.json")


if __name__ == "__main__":
    main()
