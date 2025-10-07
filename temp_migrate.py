import json
import toml  # A library for reading .toml files
from pathlib import Path
from uuid import uuid4
import chromadb.cloud
from sentence_transformers import SentenceTransformer

# --- STEP 1: Manually define your secrets here for the script ---
# You can copy these directly from your .streamlit/secrets.toml file

# ChromaDB Credentials
CHROMA_API_KEY = "your-chromadb-api-key-here"
CHROMA_TENANT = "429054f3-d864-4a24-8abd-4a0ce5088f11"
CHROMA_DATABASE = "esclation"

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
ERRORS_KB_PATH = CACHE_DIR / "gl_errors_kb.json"
PENDING_KB_PATH = CACHE_DIR / "pending_solutions.json"
LIVE_KB_COLLECTION = "live_errors_kb"
PENDING_KB_COLLECTION = "pending_errors_kb"


# --- HELPER FUNCTIONS (No Streamlit dependencies) ---
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def get_chroma_client():
    return chromadb.cloud.HttpClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )


def approve_solution_db(solution: dict, live_collection):
    solution.pop("review_id", None)
    model = get_embedding_model()
    text_to_embed = f"{solution.get('message_text', '')
    } {solution.get('reason', '')}"
    embedding = model.encode(text_to_embed).tolist()

    live_collection.add(
        ids=[str(solution["message_number"])],
        embeddings=[embedding],
        metadatas=[solution]
    )


def submit_solution_for_review_db(solution: dict, pending_collection):
    solution["review_id"] = f"review_{uuid4().hex[:8]}"
    pending_collection.add(
        ids=[solution["review_id"]],
        metadatas=[solution]
    )


# --- MIGRATION LOGIC ---
def migrate():
    print("--- Connecting to ChromaDB... ---")
    client = get_chroma_client()
    live_collection = client.get_or_create_collection(name=LIVE_KB_COLLECTION)
    pending_collection = client.get_or_create_collection(name=PENDING_KB_COLLECTION)
    print("--- Connection successful. ---")

    # Migrate Live KB
    print("\n--- Migrating LIVE Knowledge Base (gl_errors_kb.json) ---")
    if ERRORS_KB_PATH.exists():
        with open(ERRORS_KB_PATH, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        for item in kb_data:
            try:
                approve_solution_db(item, live_collection)
                print(f"  > Migrated error #{item['message_number']}")
            except Exception as e:
                print(f"  > SKIPPED #{item['message_number']}. Reason: {e}")
    else:
        print("SKIPPED: gl_errors_kb.json not found.")

    # Migrate Pending KB
    print("\n--- Migrating PENDING Solutions (pending_solutions.json) ---")
    if PENDING_KB_PATH.exists():
        with open(PENDING_KB_PATH, "r", encoding="utf-8") as f:
            pending_data = json.load(f)
        for item in pending_data:
            try:
                submit_solution_for_review_db(item, pending_collection)
                print(f"  > Migrated pending item #{item.get('message_number', 'N/A')}")
            except Exception as e:
                print(f"  > SKIPPED pending item. Reason: {e}")
    else:
        print("SKIPPED: pending_solutions.json not found.")


if __name__ == "__main__":
    # You might need to install the toml library
    # pip install toml
    migrate()
    print("\n✅ All migrations finished.")