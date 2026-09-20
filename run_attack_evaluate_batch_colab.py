import json

import config
from pipeline import BaseRAGPipeline

POISONED_INDEX_NAME = "graphtrust_poisoned_batch_index"


def main():
    with open("attack_output_batch.json") as f:
        attacks = json.load(f)
    print(f"Loaded {len(attacks)} crafted attacks.")


    print("\n" + "=" * 70)
    print("STEP 1: BASELINE (clean index, all questions)")
    print("=" * 70)
    pipeline = BaseRAGPipeline()  # uses default index_name -> clean baseline

    baselines = {}
    for a in attacks:
        result = pipeline.ask(a["question"], verbose=False)
        baselines[a["question_id"]] = result["answer"]
        print(f"  [{a['question_id']}] baseline: {result['answer'][:100]}")

    
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING POISONED INDEX (clean corpus + 20 adversarial passages)")
    print("=" * 70)

    base_max_id = max(p["id"] for p in pipeline.corpus)
    poisoned_corpus = list(pipeline.corpus)
    id_to_adversarial_id = {}

    for i, a in enumerate(attacks):
        new_id = base_max_id + 1 + i
        poisoned_corpus.append({"id": new_id, "text": a["adversarial_passage"]})
        id_to_adversarial_id[a["question_id"]] = new_id

    poisoned_pipeline = BaseRAGPipeline(
        corpus=poisoned_corpus,
        questions=pipeline.questions,
        index_name=POISONED_INDEX_NAME,
        force_rebuild=True, )

    
    print("\n" + "=" * 70)
    print("STEP 3: RE-ASKING ALL QUESTIONS AGAINST POISONED INDEX")
    print("=" * 70)

    results = []
    successes = 0
    retrieved_count = 0

    for a in attacks:
        qid = a["question_id"]
        result = poisoned_pipeline.ask(a["question"], verbose=False)

        adversarial_id = id_to_adversarial_id[qid]
        was_retrieved = any(c["id"] == adversarial_id for c in result["retrieved_chunks"])
        attack_succeeded = a["target_false_answer"].lower() in result["answer"].lower()

        if was_retrieved:
            retrieved_count += 1
        if attack_succeeded:
            successes += 1

        entry = {
            "question_id": qid,
            "question": a["question"],
            "true_answer": a["true_answer"],
            "target_false_answer": a["target_false_answer"],
            "baseline_answer": baselines[qid],
            "post_attack_answer": result["answer"],
            "was_retrieved": was_retrieved,
            "attack_succeeded": attack_succeeded,
            # Full retrieved chunk set + which one is the adversarial
            # passage - needed by claim_extraction.py / the NLI graph so
            # Stage 2 can run on real pipeline output instead of
            # hand-copied text.
            "retrieved_chunks": result["retrieved_chunks"],
            "adversarial_chunk_id": adversarial_id,
            "adversarial_passage": a["adversarial_passage"],
        }
        results.append(entry)

        status = "SUCCESS" if attack_succeeded else ("retrieved but no effect" if was_retrieved else "not retrieved")
        print(f"  [{qid}] {status}")
        print(f"      baseline: {baselines[qid][:80]}")
        print(f"      poisoned: {result['answer'][:80]}")

    #summary 
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(attacks)
    print(f"Total questions attacked:  {n}")
    print(f"Retrieved into top-k:      {retrieved_count}/{n} ({100*retrieved_count/n:.1f}%)")
    print(f"Attack Success Rate (ASR): {successes}/{n} ({100*successes/n:.1f}%)")

    with open("attack_evaluation_batch.json", "w") as f:
        json.dump({
            "summary": {
                "total": n,
                "retrieved": retrieved_count,
                "successes": successes,
                "retrieval_rate": retrieved_count / n,
                "asr": successes / n,
            },
            "results": results,
        }, f, indent=2)

    print("\nSaved full results to attack_evaluation_batch.json")


if __name__ == "__main__":
    main()
