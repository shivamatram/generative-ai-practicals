"""
Experiment 4: Prompt engineering and Retrieval Augmented Generation (RAG)
A small local RAG demo using sentence-transformers + FAISS + a Hugging Face text model.

Run:
    pip install -U sentence-transformers faiss-cpu transformers torch
    python experiment_4_rag_prompting.py

For a lightweight demo, the script uses a small FLAN-T5 model. The first run downloads
models from Hugging Face and therefore needs an internet connection.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline

DOCUMENTS = [
    "Retrieval Augmented Generation (RAG) combines information retrieval with text generation.",
    "In RAG, documents are split into chunks and converted into vector embeddings.",
    "A vector database stores embeddings and retrieves chunks similar to a user's question.",
    "The retrieved context is added to the prompt so the language model can answer using evidence.",
    "Prompt engineering means writing clear instructions and useful context for a language model.",
    "RAG can reduce hallucination, but it does not guarantee that every generated answer is correct."
]

def build_vector_index(documents, encoder):
    vectors = encoder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    return index

def retrieve(question, documents, encoder, index, top_k=3):
    query_vector = encoder.encode([question], convert_to_numpy=True,
                                  normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query_vector, min(top_k, len(documents)))
    return [documents[i] for i in ids[0] if i >= 0]

def make_prompt(question, context):
    context_text = "\n".join(f"- {item}" for item in context)
    return f"""Answer the question using only the context below.
If the answer is not present in the context, say that the context does not contain the answer.
Do not invent facts.

Context:
{context_text}

Question: {question}
Answer:"""

def main():
    print("Loading embedding model...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = build_vector_index(DOCUMENTS, encoder)

    print("Loading language model...")
    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    question = input("\nAsk a question about RAG: ").strip()
    if not question:
        question = "How does RAG help reduce hallucination?"

    context = retrieve(question, DOCUMENTS, encoder, index)
    prompt = make_prompt(question, context)
    result = generator(prompt, max_new_tokens=120, do_sample=False)

    print("\nRetrieved context:")
    for item in context:
        print("-", item)
    print("\nGenerated answer:")
    print(result[0]["generated_text"])

if __name__ == "__main__":
    main()
