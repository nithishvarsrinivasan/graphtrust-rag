import json
import itertools
import networkx as nx
import spacy
from pathlib import Path
from transformers import pipeline as hf_pipeline
import config

CONTRADICTION_THRESHOLD = 0.5
ENTAILMENT_THRESHOLD = 0.5
SUSPICION_THRESHOLD = -0.3
TOP_K = 5

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

print("Loading NLI model...")
nli_pipe = hf_pipeline(
    "text-classification",
    model=config.NLI_MODEL,
    device=-1,
    top_k=None,
)

print("Loading poisoned corpus...")
with open("poisoned_corpus.json") as f:
    poisoned_corpus = json.load(f)

injection_map = {}
if Path("injection_map.json").exists():
    with open("injection_map.json") as f:
        injection_map = json.load(f)

adversarial_ids = set(injection_map.values())
print(f"Corpus: {len(poisoned_corpus)} passages | Known adversarial ids: {len(adversarial_ids)}")

# ── load retriever 
print("Loading retriever...")
from retriever import get_or_build_retriever
retriever = get_or_build_retriever(poisoned_corpus)
print("Ready.\n")


# ── claim extraction
def _entity_density(sent) -> int:
    return len(sent.ents) * 2 + len(list(sent.noun_chunks))


def extract_claim(text: str) -> str:
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return text.strip()
    return max(sentences, key=_entity_density).text.strip()


# ── NLI pairwise 
def classify_pair(claim_a: str, claim_b: str) -> dict:
    raw = nli_pipe(f"{claim_a} [SEP] {claim_b}")[0]
    scores = {r["label"]: r["score"] for r in raw}
    return {"label": max(scores, key=scores.get), "scores": scores}


# ── graph trust scoring 
FILLER_MARKERS = [
    "encyclopedia summaries frequently identify",
    "the topic concerns the same subject",
    "commonly described in reference works as being associated with",
]

def is_filler_text(text: str) -> bool:
    return any(marker in text.lower() for marker in FILLER_MARKERS)


def compute_trust_scores(chunks: list[dict]) -> dict:
    G = nx.Graph()
    for chunk in chunks:
        G.add_node(chunk["id"])

    for i, j in itertools.combinations(range(len(chunks)), 2):
        ca = chunks[i]["claim"]
        cb = chunks[j]["claim"]
        result = classify_pair(ca, cb)
        scores = result["scores"]
        label = result["label"]

        if label == "contradiction" and scores["contradiction"] >= CONTRADICTION_THRESHOLD:
            G.add_edge(
                chunks[i]["id"], chunks[j]["id"],
                weight=-scores["contradiction"] * 2,
                relation="contradiction",
            )
        elif label == "entailment" and scores["entailment"] >= ENTAILMENT_THRESHOLD:
            G.add_edge(
                chunks[i]["id"], chunks[j]["id"],
                weight=+scores["entailment"],
                relation="entailment",
            )

    return {
        node: sum(d["weight"] for _, _, d in G.edges(node, data=True))
        for node in G.nodes()
    }, G


def flag_chunks(chunks: list[dict], trust_scores: dict, G) -> dict:
    if not chunks:
        return {}

    max_score = max(c["score"] for c in chunks)
    SCORE_RATIO_THRESHOLD = 0.60

    # Build per-node edge counts
    node_edges = {}
    for chunk in chunks:
        cid = chunk["id"]
        contradiction_count = sum(
            1 for _, _, d in G.edges(cid, data=True)
            if d["relation"] == "contradiction"
        )
        entailment_count = sum(
            1 for _, _, d in G.edges(cid, data=True)
            if d["relation"] == "entailment"
        )
        node_edges[cid] = {
            "contradiction_count": contradiction_count,
            "entailment_count": entailment_count,
        }

    # Find contradiction pairs where scores are very close
    # → flag the lower-scoring chunk in the pair
    close_score_flagged = set()
    for u, v, d in G.edges(data=True):
        if d["relation"] == "contradiction":
            u_score = next(c["score"] for c in chunks if c["id"] == u)
            v_score = next(c["score"] for c in chunks if c["id"] == v)
            score_gap = abs(u_score - v_score)
            # If scores are very close (gap < 2.0), flag the lower one
            if score_gap < 2.0:
                loser = u if u_score < v_score else v
                close_score_flagged.add(loser)

    flagged = {}
    for chunk in chunks:
        cid = chunk["id"]
        score = trust_scores.get(cid, 0.0)
        ec = node_edges[cid]

        # Signal 1: filler text — always flag
        if is_filler_text(chunk["text"]):
            is_suspicious = True

        # Signal 2: retrieval score far below top chunk
        elif chunk["score"] < max_score * SCORE_RATIO_THRESHOLD:
            is_suspicious = True

        # Signal 3: involved in contradiction with close-scoring chunk
        # → the lower scorer is the outlier
        elif cid in close_score_flagged:
            is_suspicious = True

        # Signal 4: entailment-only but has MORE entailment edges than
        # the group average → suspiciously agreeable (paraphrase attack)
        # flag only if it's a known-similar duplicate (near-zero score gap with top)
        elif (
            ec["entailment_count"] > 0
            and ec["contradiction_count"] == 0
            and chunk["score"] >= max_score * 0.95  # very close to top
            and chunk["rank"] > 1                    # but not rank 1
        ):
            # Check if rank 1 chunk has same entailment to it
            # If so, this is a near-duplicate — flag it
            rank1_id = next(c["id"] for c in chunks if c["rank"] == 1)
            if G.has_edge(cid, rank1_id):
                edge_data = G[cid][rank1_id]
                if edge_data.get("relation") == "entailment":
                    is_suspicious = True
                else:
                    is_suspicious = False
            else:
                is_suspicious = False

        else:
            is_suspicious = False

        flagged[cid] = {
            "trust_score": round(score, 4),
            "retrieval_score": round(chunk["score"], 4),
            "contradiction_count": ec["contradiction_count"],
            "entailment_count": ec["entailment_count"],
            "flag": "SUSPICIOUS" if is_suspicious else "TRUSTED",
            "is_known_adversarial": cid in adversarial_ids,
        }

    return flagged
# ── main ask function 
def ask(question: str, verbose: bool = True) -> dict:

    # 1. Retrieve from poisoned index — shape: {id, text, score, rank}
    raw_results = retriever.search(question, k=TOP_K)

    chunks = []
    for r in raw_results:
        chunks.append({
            "id": r["id"],
            "text": r["text"],
            "score": r["score"],
            "rank": r["rank"],
            "claim": extract_claim(r["text"]),
        })

    # 2. Trust scoring via NLI + graph
    trust_scores, G = compute_trust_scores(chunks)

    # 3. Flag each chunk
    flagged = flag_chunks(chunks, trust_scores, G)
    

    # 4. Build clean context from trusted chunks only
    trusted_chunks = [
        c for c in chunks if flagged[c["id"]]["flag"] == "TRUSTED"
    ]
    trusted_chunks.sort(key=lambda x: x["rank"])

    # Fallback: if everything flagged, use highest trust score chunk
    if not trusted_chunks:
        trusted_chunks = [max(chunks, key=lambda x: trust_scores.get(x["id"], 0.0))]

    clean_context = "\n\n".join(c["text"] for c in trusted_chunks)

    # 5. Generate answer from clean context
    from generator import generate_answer
    answer = generate_answer(question, trusted_chunks)

    # 6. Print defense report
    if verbose:
        print(f"\n{'='*65}")
        print(f"Q: {question}")
        print(f"\n── Retrieved chunks (sorted by trust score) ──")
        for chunk in sorted(chunks, key=lambda x: trust_scores.get(x["id"], 0.0)):
            cid = chunk["id"]
            f = flagged[cid]
            adv_tag = " [KNOWN ADV]" if f["is_known_adversarial"] else ""
            sus_tag = " ← FLAGGED" if f["flag"] == "SUSPICIOUS" else ""
            print(
                f"  [rank {chunk['rank']}] id={cid:<5} | "
                f"ret={f['retrieval_score']:>6.3f} | "
                f"trust={f['trust_score']:>7.4f} | "
                f"contra={f['contradiction_count']} entail={f['entailment_count']} | "
                f"{f['flag']}{sus_tag}{adv_tag}"
            )
            print(f"    claim: {chunk['claim'][:110]}")

        print(f"\n── Trusted chunks used for answer: {len(trusted_chunks)} ──")
        print(f"── Suspicious chunks blocked:      "
              f"{sum(1 for v in flagged.values() if v['flag'] == 'SUSPICIOUS')} ──")
        print(f"\n── Answer ──")
        print(f"  {answer}")
        print(f"{'='*65}\n")

    # 7. Build and return result dict
    adv_in_retrieved = [c for c in chunks if c["id"] in adversarial_ids]
    adv_correctly_flagged = [
        c for c in adv_in_retrieved
        if flagged[c["id"]]["flag"] == "SUSPICIOUS"
    ]

    return {
        "question": question,
        "answer": answer,
        "chunks": [
            {
                "id": c["id"],
                "rank": c["rank"],
                "trust_score": flagged[c["id"]]["trust_score"],
                "flag": flagged[c["id"]]["flag"],
                "is_known_adversarial": flagged[c["id"]]["is_known_adversarial"],
                "claim": c["claim"],
                "text": c["text"],
            }
            for c in chunks
        ],
        "stats": {
            "trusted": len(trusted_chunks),
            "suspicious": sum(1 for v in flagged.values() if v["flag"] == "SUSPICIOUS"),
            "adversarial_retrieved": len(adv_in_retrieved),
            "adversarial_flagged": len(adv_correctly_flagged),
        }
    }

def run_batch_evaluation(
    input_path="claim_extraction_output.json",
    output_path="defense_results.json",
):
    """
    Runs live defense on all questions from claim_extraction_output.json
    and writes defense_results.json with final_answer included.
    Used for evaluation metrics.
    """
    data = json.loads(Path(input_path).read_text())
    results = []

    for entry in data:
        qid = entry["question_id"]
        question = entry["question"]
        true_answer = entry["true_answer"]
        target_false_answer = entry["target_false_answer"]
        adv_id = entry["adversarial_chunk_id"]

        print(f"\n[eval] {qid[:20]}... | Q: {question[:60]}")

        result = ask(question, verbose=True)

        # Map chunk results to evaluation format
        chunk_results = []
        for c in result["chunks"]:
            chunk_results.append({
                "chunk_id": c["id"],
                "trust_score": c["trust_score"],
                "flag": c["flag"],
                "is_adversarial": c["id"] == adv_id,
            })

        adv_chunk = next((c for c in chunk_results if c["is_adversarial"]), None)
        adv_flagged = adv_chunk["flag"] == "SUSPICIOUS" if adv_chunk else False

        results.append({
            "question_id": qid,
            "question": question,
            "true_answer": true_answer,
            "target_false_answer": target_false_answer,
            "adversarial_chunk_id": adv_id,
            "adversarial_chunk_flagged": adv_flagged,
            "final_answer": result["answer"],        # ← what evaluate.py needs
            "chunk_results": chunk_results,
            "stats": result["stats"],
        })

    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\nBatch evaluation done → {output_path}")
    return results

# ── interactive loop 
if __name__ == "__main__":
    print("GraphTrust-RAG — Live Defense")
    print("Ctrl+C to exit\n")
    while True:
        try:
            q = input("Question: ").strip()
            if q:
                ask(q)
        except KeyboardInterrupt:
            print("\nDone.")
            break
