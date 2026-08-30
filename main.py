"""
main.py
-------
Extracted from your notebook cells 31-32, unchanged.
"""

from pipeline import BaseRAGPipeline


def main():
    print("Loading pipeline")
    pipeline = BaseRAGPipeline()

    print("\nType a question (or 'quit' to exit).")
    print(f"  {pipeline.questions[0]['question']}")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        pipeline.ask(question)


if __name__ == "__main__":
    main()
