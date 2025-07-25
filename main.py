PDF_DIR = "./pdfs"

import os
import fitz  # PyMuPDF
import concurrent.futures
from multiprocessing import cpu_count
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_file):
    """Extracts text from a single PDF file using PyMuPDF."""
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    with fitz.open(pdf_path) as pdf:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            page_texts = list(executor.map(lambda page: page.get_text("text"), pdf))
    text = "\n".join(filter(None, page_texts))  # Remove empty values
    return {"filename": pdf_file, "text": text}

def extract_text_from_pdfs():
    """Extracts text from all PDFs in the directory using multiprocessing."""
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        return list(executor.map(extract_text_from_pdf, pdf_files))

def generate_embeddings(chunked_docs, batch_size=32):
    """Generate embeddings with progress tracking and batch processing."""
    model = SentenceTransformer("intfloat/e5-base")
    all_embeddings = []

    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i:i + batch_size]
        batch_texts = [doc["text"] for doc in batch]
        print(f"Processing batch {i // batch_size + 1}/{(len(chunked_docs) + batch_size - 1) // batch_size}")
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=True)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def setup_database():
    """Loads PDFs, splits text, encodes, and stores embeddings in ChromaDB."""
    # Check for cached embeddings
    if os.path.exists("./icpc_embeddings.npy") and os.path.exists("./icpc_chunks.npy"):
        print("Loading existing embeddings...")
        all_embeddings = np.load("./icpc_embeddings.npy", allow_pickle=True)
        chunked_docs = np.load("./icpc_chunks.npy", allow_pickle=True)
    else:
        # Extract and split text
        documents = extract_text_from_pdfs()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunked_docs = [
            {"text": chunk, "source": doc["filename"], "chunk_id": i}
            for doc in documents for i, chunk in enumerate(text_splitter.split_text(doc["text"]))
        ]

        # Generate embeddings in batches
        all_embeddings = generate_embeddings(chunked_docs)

        # Save to disk
        np.save("./icpc_embeddings.npy", np.array(all_embeddings, dtype=object))
        np.save("./icpc_chunks.npy", np.array(chunked_docs, dtype=object))

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection("icpc")

    # Store embeddings in batches (to avoid memory issues)
    batch_size = 100
    for i in range(0, len(chunked_docs), batch_size):
        batch_docs = chunked_docs[i:i + batch_size]
        batch_embeddings = all_embeddings[i:i + batch_size]

        collection.add(
            ids=[f"{doc['source']}_chunk_{doc['chunk_id']}" for doc in batch_docs],
            embeddings=[emb.tolist() for emb in batch_embeddings],
            metadatas=[{"source": doc["source"], "chunk_id": doc["chunk_id"]} for doc in batch_docs],
            documents=[doc["text"] for doc in batch_docs]
        )

def retrieve_context(query, top_k=5):
    encoder = SentenceTransformer("intfloat/e5-base")
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_collection("icpc")
    query_embedding = encoder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]
    return "\n\n".join(retrieved_docs)

def generate_answer(query):
    context = retrieve_context(query)

    prompt = f"""
    You are a helpful and accurate assistant.
    You will be given content extracted from a website (including multiple pages). Use this information to answer questions users might ask about the website. If the answer is not present in the provided content, respond with “I don't know based on the website data.”

    Website Domain: https://amritaicpc.in

    Instructions:
        Stick strictly to the provided context.
        If a question asks about contest rules, regions, results, registration, team formation, or ICPC history, extract the answer from the relevant section.
        Do not guess or hallucinate details.
        Answer clearly and concisely.
        If the answer is not in the context, say "This is out of scope for this chatbot."

    Website Content:
    {context}

    User Question:
    {query}

    Your Answer:
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    setup_database()  # Caches on first run

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        print("\nSearching for relevant text...")
        print(generate_answer(question))
