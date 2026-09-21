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
                weight=-scores["contradiction"],
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
    }


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
    trust_scores = compute_trust_scores(chunks)

    # 3. Flag each chunk
    flagged = {}
    for chunk in chunks:
        cid = chunk["id"]
        score = trust_scores.get(cid, 0.0)
        flagged[cid] = {
            "trust_score": round(score, 4),
            "flag": "SUSPICIOUS" if score < SUSPICION_THRESHOLD else "TRUSTED",
            "is_known_adversarial": cid in adversarial_ids,
        }

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
    answer = generate_answer(question, clean_context)

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
                f"trust={f['trust_score']:>7.4f} | "
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
