import os

import config


def _collection_from_corpus(corpus):
    texts = [p["text"] for p in corpus]
    id_map = {i: corpus[i]["id"] for i in range(len(corpus))}
    return texts, id_map


def index_exists(index_name=None):
    index_name = index_name or config.COLBERT_INDEX_NAME
    index_path = os.path.join(config.COLBERT_INDEX_ROOT, index_name)
    return os.path.exists(index_path)


def build_index(corpus, index_name=None):
    index_name = index_name or config.COLBERT_INDEX_NAME

    print("Loading Indexer...")
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig

    texts, id_map = _collection_from_corpus(corpus)

    with Run().context(RunConfig(nranks=1, experiment="graphtrust", avoid_fork_if_possible=True)):
        cfg = ColBERTConfig(root=config.COLBERT_INDEX_ROOT)
        print("Creating Indexer...")
        indexer = Indexer(checkpoint=config.COLBERT_CHECKPOINT, config=cfg)
        print("Starting indexing...")
        indexer.index(name=index_name, collection=texts, overwrite=True)
        print("Finished indexing.")
    print(f"Indexed {len(texts)} passages as '{index_name}'.")
    return id_map


class ColBERTRetriever:
    def __init__(self, corpus, id_map=None, index_name=None):
        from colbert import Searcher
        from colbert.infra import Run, RunConfig, ColBERTConfig

        index_name = index_name or config.COLBERT_INDEX_NAME

        self.corpus = corpus
        self.corpus_by_position = {i: corpus[i] for i in range(len(corpus))}
        self.id_map = id_map or {i: corpus[i]["id"] for i in range(len(corpus))}

        with Run().context(RunConfig(nranks=1, experiment="graphtrust", avoid_fork_if_possible=True)):
            cfg = ColBERTConfig(root=config.COLBERT_INDEX_ROOT)
            self.searcher = Searcher(index=index_name, config=cfg)

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


def get_or_build_retriever(corpus, index_name=None, force_rebuild=False):
    """
    index_name lets you keep multiple indexes on disk side by side (e.g.
    a clean baseline index and a separately-named poisoned index) instead
    of the poisoned build silently overwriting your baseline.
    force_rebuild=True skips the "reuse if exists" check - use this when
    you've changed the corpus content but kept the same index_name.
    """
    index_name = index_name or config.COLBERT_INDEX_NAME
    id_map = {i: corpus[i]["id"] for i in range(len(corpus))}
    if force_rebuild or not index_exists(index_name):
        id_map = build_index(corpus, index_name=index_name)
    return ColBERTRetriever(corpus, id_map=id_map, index_name=index_name)


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
