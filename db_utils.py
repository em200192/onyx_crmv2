from uuid import uuid4

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import gspread
import re

from datetime import datetime, timezone

# --- CONSTANTS ---
LIVE_KB_COLLECTION = "live_errors_kb"
PENDING_KB_COLLECTION = "pending_errors_kb"


# --- EMBEDDING MODEL (Cached) ---
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# --- CHROMA DB CLIENT ---
# In db_utils.py

# Add this new import at the top of the file

from chromadb.config import Settings

# In db_utils.py
@st.cache_resource
def get_chroma_client():
    """Initializes the ChromaDB Cloud client."""
    return chromadb.CloudClient(
        api_key=st.secrets["CHROMA_API_KEY"],
        tenant=st.secrets["CHROMA_TENANT"],
        database=st.secrets["CHROMA_DATABASE"]
    )


@st.cache_resource
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name='paraphrase-multilingual-MiniLM-L12-v2')


def get_collections():
    client = get_chroma_client()
    embedding_func = get_embedding_function()
    live_collection = client.get_or_create_collection(name=LIVE_KB_COLLECTION, embedding_function=embedding_func)
    pending_collection = client.get_or_create_collection(name=PENDING_KB_COLLECTION, embedding_function=embedding_func)
    return live_collection, pending_collection


def get_live_kb_collection():
    client = get_chroma_client()
    embedding_func = get_embedding_function()
    return client.get_or_create_collection(name=LIVE_KB_COLLECTION, embedding_function=embedding_func)

def get_pending_kb_collection():
    client = get_chroma_client()
    embedding_func = get_embedding_function()
    return client.get_or_create_collection(name=PENDING_KB_COLLECTION, embedding_function=embedding_func)


# --- GOOGLE SHEETS CLIENT ---
@st.cache_resource
def get_gsheet():
    gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
    sheet = gc.open(st.secrets["GSHEET_NAME"]).sheet1
    if not sheet.get_all_values():
        sheet.append_row(["ticket_id", "timestamp_utc", "query", "reason", "status"])
    return sheet



# --- DATABASE FUNCTIONS ---
def submit_solution_for_review_db(new_entry: dict):
    """Adds or UPDATES a solution in the pending review collection using upsert."""
    pending_collection = get_pending_kb_collection()

    # Use the existing review_id if this is an update, or create a new one if it's a new entry.
    review_id = new_entry.get("review_id") or f"review_{uuid4().hex[:8]}"

    document_text = f"Error {new_entry.get('message_number', '')}: {new_entry.get('message_text', '')}"
    clean_metadata = {k: v for k, v in new_entry.items() if v is not None}
    # Ensure review_id is in the metadata for consistency
    clean_metadata['review_id'] = review_id

    # --- THIS IS THE FIX ---
    # Use upsert() to either create a new entry or update an existing one based on the ID.
    pending_collection.upsert(
        ids=[review_id],
        metadatas=[clean_metadata],
        documents=[document_text]
    )
def approve_solution_db(solution: dict):
    live_collection, pending_collection = get_collections()
    review_id = solution.pop("review_id", None)

    model = get_embedding_model()
    text_to_embed = f"{solution.get('message_text', '')} {solution.get('reason', '')}"
    embedding = model.encode(text_to_embed).tolist()

    live_collection.add(
        ids=[str(solution["message_number"])],
        embeddings=[embedding],
        metadatas=[solution]
    )
    if review_id:
        pending_collection.delete(ids=[review_id])


def reject_solution_db(review_id: str):
    _, pending_collection = get_collections()
    pending_collection.delete(ids=[review_id])


def search_errors_db(query: str, n_results: int = 1):
    """
    Searches the ChromaDB collection. First tries to extract and find a
    numeric ID, then falls back to a semantic search.
    """
    live_collection = get_live_kb_collection()
    cleaned_query = query.strip()

    # --- NEW: Smart number extraction ---
    # First, try to find a 3+ digit number within the query string.
    numbers_found = re.findall(r'\d{3,}', cleaned_query)
    if numbers_found:
        number_id = numbers_found[0]
        # Try a direct lookup using the found number as an ID.
        result = live_collection.get(ids=[number_id], include=["metadatas"])
        if result and result.get('ids'):
            print(f"--- DEBUG: Found direct match for ID '{number_id}' in query. ---")
            # Format the result to match the structure of a query() result
            return {"metadatas": [result['metadatas']], "distances": [[0.0]]}
    # --- END of new logic ---

    # If no number was found, or if the direct ID lookup failed, perform a full semantic search.
    print(f"--- DEBUG: No direct ID match found. Performing semantic search for '{query}'. ---")
    results = live_collection.query(query_texts=[query], n_results=n_results)

    return results if results and results.get('ids') and results['ids'][0] else None
# In db_utils.py

def load_pending_solutions_db():
    """Fetches all items from the pending collection, returning both IDs and metadata."""
    pending_collection = get_pending_kb_collection()
    results = pending_collection.get()
    # Return a list of (id, metadata) tuples
    if results and results.get('ids'):
        return list(zip(results['ids'], results.get('metadatas', [])))
    return []


def load_all_errors_db():
    """Fetches all items from the live KB collection."""
    live_collection = get_live_kb_collection()
    # The include parameter ensures we get everything we need.
    results = live_collection.get(include=["metadatas"])
    return results.get('metadatas', [])
def log_escalation_gsheet(ticket_id: str, query: str, reason: str):
    sheet = get_gsheet()
    sheet.append_row([ticket_id, datetime.now(timezone.utc).isoformat(), query, reason, "pending"])


def load_conversation_history_from_gcs(ticket_id: str):
    """Downloads and parses a chat history JSON file from GCS."""
    if not ticket_id:
        return None
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import json  # Make sure json is imported

        creds_info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_info)
        client = storage.Client(credentials=creds, project=creds.project_id)

        bucket_name = st.secrets["gcs"]["bucket_name"]
        if not bucket_name: return None

        bucket = client.bucket(bucket_name)
        object_name = f"escalation_logs/{ticket_id}.json"

        blob = bucket.blob(object_name)
        if blob.exists():
            history_json = blob.download_as_string()
            return json.loads(history_json)
        else:
            print(f"History file not found in GCS: {object_name}")
            return None

    except Exception as e:
        st.error(f"Failed to load chat history from GCS: {e}")
        return None

def load_escalations_gsheet():
    sheet = get_gsheet()
    return sheet.get_all_records()


def mark_escalation_as_done_gsheet(ticket_id: str):
    sheet = get_gsheet()
    cell = sheet.find(ticket_id)
    if cell:
        sheet.update_cell(cell.row, 5, "done")  # Column 5 is 'status'
