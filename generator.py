"""
generator.py
------------
Extracted from your notebook cells 25-26, unchanged - this is the
version with the reasoning-effort cap and empty-content diagnostic you
already fixed.
"""

from openai import OpenAI

import config

_client = None


def get_client():
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. In Colab, set it via "
                "google.colab.userdata in your setup cell; locally, set "
                "it as a plain environment variable before running."
            )
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
    return _client


RAG_SYSTEM_PROMPT = """You are a question-answering assistant. Answer the \
user's question using ONLY the information in the provided context. \
If the context does not contain enough information to answer, say so \
explicitly rather than guessing. Keep answers concise (1-2 sentences)."""


def build_rag_prompt(question, retrieved_chunks):
    context_block = "\n\n".join(
        f"[Passage {i+1}]\n{chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer(question, retrieved_chunks):
    client = get_client()
    user_prompt = build_rag_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=config.GENERATOR_MODEL,
        max_tokens=config.GENERATOR_MAX_TOKENS,
        temperature=config.GENERATOR_TEMPERATURE,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"reasoning": {"effort": "low"}},
    )

    message = response.choices[0].message
    content = (message.content or "").strip()

    if not content:
        finish_reason = response.choices[0].finish_reason
        reasoning_text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)

        print(f"[generator] WARNING: empty content. finish_reason={finish_reason!r}")
        if reasoning_text:
            print(f"[generator] Model was still reasoning when cut off "
                  f"(first 200 chars): {reasoning_text[:200]}...")

        raise RuntimeError(
            f"Nemotron returned empty content (finish_reason={finish_reason}). "
            f"Try increasing GENERATOR_MAX_TOKENS (currently "
            f"{config.GENERATOR_MAX_TOKENS})."
        )

    return content


if __name__ == "__main__":
    dummy_chunks = [
        {"text": "The Eiffel Tower is located in Paris, France, and was "
                  "completed in 1889."},
    ]
    answer = generate_answer("Where is the Eiffel Tower?", dummy_chunks)
    print("Answer:", answer)
