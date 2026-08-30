"""
pipeline.py
-----------
Extracted from your notebook cells 28-29, unchanged.
"""

import data_loader
import generator
import retriever


class BaseRAGPipeline:
    def __init__(self, corpus=None, questions=None):
        if corpus is None or questions is None:
            corpus, questions = data_loader.prepare_corpus_and_questions()
        self.corpus = corpus
        self.questions = questions
        self.retriever = retriever.get_or_build_retriever(corpus)

    def retrieve(self, question, k=None):
        return self.retriever.search(question, k=k)

    def generate(self, question, retrieved_chunks):
        return generator.generate_answer(question, retrieved_chunks)

    def ask(self, question, k=None, verbose=True):
        chunks = self.retrieve(question, k=k)
        answer = self.generate(question, chunks)

        result = {
            "question": question,
            "retrieved_chunks": chunks,
            "answer": answer,
        }

        if verbose:
            print(f"\nQ: {question}")
            print("Retrieved chunks:")
            for c in chunks:
                print(f"  [score={c['score']:.3f}] {c['text'][:120]}...")
            print(f"A: {answer}")

        return result


if __name__ == "__main__":
    pipeline = BaseRAGPipeline()
    sample_question = pipeline.questions[0]["question"]
    expected_answer = pipeline.questions[0]["answer"]

    result = pipeline.ask(sample_question)
    print(f"\n(Expected answer from dataset: {expected_answer})")
