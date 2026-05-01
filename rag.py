import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from pii import mask_pii

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# -----------------------------
# PRIVACY BUCKETING
# -----------------------------
def income_bucket(value):
    value = float(value)
    if value < 2000:
        return "low income"
    elif value < 5000:
        return "medium income"
    return "high income"

def debt_bucket(value):
    value = float(value)
    if value < 3000:
        return "low debt"
    elif value < 7000:
        return "medium debt"
    return "high debt"

# -----------------------------
# ROW → ANONYMIZED TEXT
# -----------------------------
def row_to_text(row):
    return f"""
    Person aged {row['age']} working as {row['occupation']} in {row['city']}.
    Income level: {income_bucket(row['monthly_income'])}.
    Savings rate: {row['savings_rate']} percent.
    Debt level: {debt_bucket(row['debt'])}.
    Financial health score: {row['financial_health_score']}.
    """

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=100):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# CHROMADB
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="privacy_finance_rag")

print("Collection count:", collection.count())

# -----------------------------
# INGESTION (FIXED SAFE BATCH)
# -----------------------------
if collection.count() == 0:
    print("🔄 Creating embeddings...")

    data = pd.read_csv('data/finance_data.csv')

    documents, ids, metadatas = [], [], []

    for i, row in data.iterrows():
        text = clean_text(row_to_text(row))
        chunks = chunk_text(text)

        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{i}_{j}")

            # 🔐 FIX: metadata MUST NOT be empty
            metadatas.append({
                "type": "financial_profile",
                "privacy": "anonymized"
            })

    embeddings = model.encode(documents, batch_size=16, show_progress_bar=True)

    # 🔥 FIX: SAFE BATCH INSERT (ChromaDB limit fix)
    BATCH_SIZE = 300

    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            embeddings=[e.tolist() for e in embeddings[i:i+BATCH_SIZE]],
            ids=ids[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE]
        )

    print("✅ Data stored successfully (privacy-preserving)")
else:
    print("✅ Loaded existing DB")

# -----------------------------
# QUERY SAFETY
# -----------------------------
def query_chromadb(query):
    query = clean_text(mask_pii(query))
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    return " ".join(results['documents'][0])

# -----------------------------
# RULE ENGINE
# -----------------------------
def extract_salary(query):
    match = re.search(r'\d+', query)
    return float(match.group()) if match else None

def simple_finance_rule(query):
    query = query.lower()
    salary = extract_salary(query)

    if not salary:
        return None

    rules = {
        "food": 0.15,
        "house": 0.30,
        "housing": 0.30,
        "savings": 0.20,
        "transport": 0.10
    }

    for key, percent in rules.items():
        if key in query:
            return f"Estimated {key} expense: ${salary * percent:.2f}"

    return None

# -----------------------------
# LLM CALL
# -----------------------------
def ask_llm(query, context):

    prompt_text = f"""
You are a privacy-preserving financial assistant.

RULES:
- Never reveal sensitive personal data
- Use only generalized information
- Keep answers short (1–2 lines)
- If unsure, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 100
                }
            }
        )

        return response.json().get("response", "Error")

    except Exception as e:
        return str(e)

# -----------------------------
# PIPELINE
# -----------------------------
def rag_pipeline(query):

    masked_query = mask_pii(query)

    # 1. RULE ENGINE FIRST
    rule_answer = simple_finance_rule(masked_query)
    if rule_answer:
        return rule_answer

    # 2. RAG
    context = query_chromadb(masked_query)

    if len(context.strip()) < 20:
        return "I don't know"

    # 3. LLM
    return ask_llm(masked_query, context)

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\n💬 Ask: ")
        if q.lower() == "exit":
            break

        print("\n🤖", rag_pipeline(q))