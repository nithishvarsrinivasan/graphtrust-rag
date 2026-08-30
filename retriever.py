import os

import config


def _collection_from_corpus(corpus):
    texts = [p["text"] for p in corpus]
    id_map = {i: corpus[i]["id"] for i in range(len(corpus))}
    return texts, id_map


def index_exists():
    index_path = os.path.join(config.COLBERT_INDEX_ROOT, config.COLBERT_INDEX_NAME)
    return os.path.exists(index_path)


def build_index(corpus):
    print("Loading Indexer...")
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig

    texts, id_map = _collection_from_corpus(corpus)

    with Run().context(RunConfig(nranks=1, experiment="graphtrust", avoid_fork_if_possible=True)):
        cfg = ColBERTConfig(root=config.COLBERT_INDEX_ROOT)
        print("Creating Indexer...")
        indexer = Indexer(checkpoint=config.COLBERT_CHECKPOINT, config=cfg)
        print("Starting indexing...")
        indexer.index(name=config.COLBERT_INDEX_NAME, collection=texts, overwrite=True)
        print("Finished indexing.")
    print(f"Indexed {len(texts)} passages as '{config.COLBERT_INDEX_NAME}'.")
    return id_map


class ColBERTRetriever:
    def __init__(self, corpus, id_map=None):
        from colbert import Searcher
        from colbert.infra import Run, RunConfig, ColBERTConfig

        self.corpus = corpus
        self.corpus_by_position = {i: corpus[i] for i in range(len(corpus))}
        self.id_map = id_map or {i: corpus[i]["id"] for i in range(len(corpus))}

        with Run().context(RunConfig(nranks=1, experiment="graphtrust", avoid_fork_if_possible=True)):
            cfg = ColBERTConfig(root=config.COLBERT_INDEX_ROOT)
            self.searcher = Searcher(index=config.COLBERT_INDEX_NAME, config=cfg)

    def search(self, query, k=None):
        k = k or config.TOP_K
        positional_ids, ranks, scores = self.searcher.search(query, k=k)

        results = []
        for pos_id, rank, score in zip(positional_ids, ranks, scores):
            passage = self.corpus_by_position[pos_id]
            results.append({
                "id": passage["id"],
                "text": passage["text"],
                "score": float(score),
                "rank": int(rank),
            })
        return results


def get_or_build_retriever(corpus):
    id_map = {i: corpus[i]["id"] for i in range(len(corpus))}
    if not index_exists():
        id_map = build_index(corpus)
    return ColBERTRetriever(corpus, id_map=id_map)


if __name__ == "__main__":
    import data_loader

    corpus, questions = data_loader.prepare_corpus_and_questions()
    print("Building index...")
    retriever = get_or_build_retriever(corpus)
    print("Index ready.")

    sample_q = questions[0]["question"]
    print(f"\nQuery: {sample_q}")
    for r in retriever.search(sample_q):
        print(f"  [{r['rank']}] score={r['score']:.3f}  {r['text'][:100]}...")
