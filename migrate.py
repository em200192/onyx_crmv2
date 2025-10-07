#!/usr/bin/env python
import os
import sys
import json
import argparse
from pathlib import Path
from uuid import uuid4

import toml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import chromadb.utils.embedding_functions
# --------------------
# CONFIG & CONSTANTS
# --------------------
BASE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = BASE_DIR / "cache"
ERRORS_KB_PATH = CACHE_DIR / "gl_errors_kb.json"
PENDING_KB_PATH = CACHE_DIR / "pending_solutions.json"

LIVE_KB_COLLECTION = "live_errors_kb"
PENDING_KB_COLLECTION = "pending_errors_kb"

CHROMA_HOST_DEFAULT = "api.trychroma.com"
BATCH_SIZE = 128  # tune as needed

# --------------------
# SECRET LOADING
# --------------------
def load_secrets():
    """Return dict with CHROMA_* from env or .streamlit/secrets.toml."""
    secrets = {
        "CHROMA_API_KEY": os.getenv("CHROMA_API_KEY"),
        "CHROMA_TENANT": os.getenv("CHROMA_TENANT"),
        "CHROMA_DATABASE": os.getenv("CHROMA_DATABASE"),
        "CHROMA_HOST": os.getenv("CHROMA_HOST", CHROMA_HOST_DEFAULT),
    }

    if all(secrets[k] for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")):
        return secrets

    # Fallback to local secrets.toml
    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        file_secrets = toml.load(secrets_path)
        for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"):
            secrets[k] = secrets[k] or file_secrets.get(k)
        secrets["CHROMA_HOST"] = secrets["CHROMA_HOST"] or file_secrets.get("CHROMA_HOST", CHROMA_HOST_DEFAULT)

    missing = [k for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE") if not secrets.get(k)]
    if missing:
        raise SystemExit(
            "Missing required secrets: "
            + ", ".join(missing)
            + "\nProvide via environment or .streamlit/secrets.toml"
        )
    return secrets

# --------------------
# CLIENT & MODEL
# --------------------
_model = None
_client = None

def get_embedding_model():
    global _model
    if _model is None:
        # multilingual, small, fast
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

# In migrate.py
def get_chroma_client(secrets):
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            api_key=secrets["CHROMA_API_KEY"],
            tenant=secrets["CHROMA_TENANT"],
            database=secrets["CHROMA_DATABASE"]
        )
    return _client
# --------------------
# UTIL
# --------------------
def chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def coerce_metadata(obj):
    """
    Ensure metadata is JSON-serializable and simple (Chroma prefers flat types).
    - Remove None keys
    - Convert non-basic types to strings
    """
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

# --------------------
# MIGRATION
# --------------------
def migrate_live(client, model):
    total = 0
    ok = 0
    skipped = 0

    if not ERRORS_KB_PATH.exists():
        print("SKIPPED: gl_errors_kb.json not found.")
        return (ok, skipped, total)

    live = client.get_or_create_collection(name=LIVE_KB_COLLECTION)

    with open(ERRORS_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Found {total} LIVE items. Embedding & upserting in batches of {BATCH_SIZE}...")

    for batch in chunked(data, BATCH_SIZE):
        ids, embeddings, metadatas = [], [], []
        texts = []

        for item in batch:
            try:
                # sanitize / prepare
                item = dict(item)
                item.pop("embedding", None)
                item.pop("review_id", None)  # remove noisy field if present

                msg_no = item.get("message_number")
                if msg_no is None:
                    raise ValueError("missing message_number")

                text = f"{item.get('message_text', '')} {item.get('reason', '')}".strip()
                if not text:
                    # minimally embed something stable to keep record
                    text = f"Message #{msg_no}"

                texts.append(text)
                ids.append(str(msg_no))
                metadatas.append(coerce_metadata(item))
            except Exception as e:
                skipped += 1
                print(f"  > SKIPPED #{item.get('message_number', 'N/A')}: {e}")

        if not ids:
            continue

        try:
            vecs = model.encode(texts, convert_to_numpy=True).tolist()
            embeddings.extend(vecs)

            # upsert = create or replace existing ids
            live.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            ok += len(ids)
            print(f"  > Upserted {len(ids)} items (last id: {ids[-1]})")
        except Exception as e:
            skipped += len(ids)
            print(f"  > BATCH FAILED ({len(ids)} items). Reason: {e}")

    return (ok, skipped, total)

def migrate_pending(client):
    total = 0
    ok = 0
    skipped = 0

    if not PENDING_KB_PATH.exists():
        print("SKIPPED: pending_solutions.json not found.")
        return (ok, skipped, total)

    pending = client.get_or_create_collection(name=PENDING_KB_COLLECTION)

    with open(PENDING_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Found {total} PENDING items. Upserting (no embeddings)...")

    for batch in chunked(data, BATCH_SIZE):
        ids, metadatas = [], []
        for item in batch:
            try:
                review_id = item.get("review_id") or f"review_{uuid4().hex[:8]}"
                ids.append(str(review_id))
                metadatas.append(coerce_metadata(item))
            except Exception as e:
                skipped += 1
                print(f"  > SKIPPED pending item: {e}")

        if not ids:
            continue

        try:
            # upsert without embeddings is fine for metadata-only
            pending.upsert(ids=ids, metadatas=metadatas)
            ok += len(ids)
            print(f"  > Upserted {len(ids)} pending items (last id: {ids[-1]})")
        except Exception as e:
            skipped += len(ids)
            print(f"  > PENDING BATCH FAILED ({len(ids)} items). Reason: {e}")

    return (ok, skipped, total)

# --------------------
# MAIN
# --------------------
def main():
    parser = argparse.ArgumentParser(description="Migrate local KB JSON files to Chroma Cloud.")
    parser.add_argument("--host", default=os.getenv("CHROMA_HOST", CHROMA_HOST_DEFAULT), help="Chroma host")
    args = parser.parse_args()

    secrets = load_secrets()
    secrets["CHROMA_HOST"] = args.host or secrets.get("CHROMA_HOST", CHROMA_HOST_DEFAULT)

    print("--- Connecting to ChromaDB Cloud ---")
    client = get_chroma_client(secrets)
    model = get_embedding_model()
    print(f"--- Connected. Tenant={secrets['CHROMA_TENANT']} Database={secrets['CHROMA_DATABASE']} Host={secrets['CHROMA_HOST']} ---")

    print("\n--- Migrating LIVE Knowledge Base ---")
    ok_l, sk_l, tot_l = migrate_live(client, model)
    print(f"LIVE: {ok_l}/{tot_l} upserted, {sk_l} skipped.")

    print("\n--- Migrating PENDING Solutions ---")
    ok_p, sk_p, tot_p = migrate_pending(client)
    print(f"PENDING: {ok_p}/{tot_p} upserted, {sk_p} skipped.")

    print("\n✅ All migrations finished.")

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
import os
import sys
import json
import argparse
from pathlib import Path
from uuid import uuid4

import toml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --------------------
# CONFIG & CONSTANTS
# --------------------
BASE_DIR = Path(__file__).parent.resolve()
CACHE_DIR = BASE_DIR / "cache"
ERRORS_KB_PATH = CACHE_DIR / "gl_errors_kb.json"
PENDING_KB_PATH = CACHE_DIR / "pending_solutions.json"

LIVE_KB_COLLECTION = "live_errors_kb"
PENDING_KB_COLLECTION = "pending_errors_kb"

CHROMA_HOST_DEFAULT = "api.trychroma.com"
BATCH_SIZE = 128  # tune as needed

# --------------------
# SECRET LOADING
# --------------------
def load_secrets():
    """Return dict with CHROMA_* from env or .streamlit/secrets.toml."""
    secrets = {
        "CHROMA_API_KEY": os.getenv("CHROMA_API_KEY"),
        "CHROMA_TENANT": os.getenv("CHROMA_TENANT"),
        "CHROMA_DATABASE": os.getenv("CHROMA_DATABASE"),
        "CHROMA_HOST": os.getenv("CHROMA_HOST", CHROMA_HOST_DEFAULT),
    }

    if all(secrets[k] for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")):
        return secrets

    # Fallback to local secrets.toml
    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        file_secrets = toml.load(secrets_path)
        for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE"):
            secrets[k] = secrets[k] or file_secrets.get(k)
        secrets["CHROMA_HOST"] = secrets["CHROMA_HOST"] or file_secrets.get("CHROMA_HOST", CHROMA_HOST_DEFAULT)

    missing = [k for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE") if not secrets.get(k)]
    if missing:
        raise SystemExit(
            "Missing required secrets: "
            + ", ".join(missing)
            + "\nProvide via environment or .streamlit/secrets.toml"
        )
    return secrets

# --------------------
# CLIENT & MODEL
# --------------------
_model = None
_client = None

def get_embedding_model():
    global _model
    if _model is None:
        # multilingual, small, fast
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

def get_chroma_client(secrets):
    global _client
    if _client is None:
        _client = chromadb.Client(
            Settings(
                chroma_api_impl="chromadb.api.cloud",
                chroma_server_host=secrets["CHROMA_HOST"],
                chroma_server_ssl=True,
                api_key=secrets["CHROMA_API_KEY"],
                tenant=secrets["CHROMA_TENANT"],
                database=secrets["CHROMA_DATABASE"],
            )
        )
    return _client

# --------------------
# UTIL
# --------------------
def chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def coerce_metadata(obj):
    """
    Ensure metadata is JSON-serializable and simple (Chroma prefers flat types).
    - Remove None keys
    - Convert non-basic types to strings
    """
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

# --------------------
# MIGRATION
# --------------------
def migrate_live(live_collection, model):
    """Migrates data from gl_errors_kb.json to the live ChromaDB collection."""
    ok, skipped, total = 0, 0, 0

    if not ERRORS_KB_PATH.exists():
        print("SKIPPED: gl_errors_kb.json not found.")
        return (ok, skipped, total)

    with open(ERRORS_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Found {total} LIVE items. Embedding & upserting in batches of {BATCH_SIZE}...")

    for batch in chunked(data, BATCH_SIZE):
        ids, embeddings, metadatas, texts = [], [], [], []

        for item in batch:
            try:
                item = dict(item)
                # This is the critical fix to prevent the quota error
                item.pop("embedding", None)

                msg_no = item.get("message_number")
                if not msg_no:
                    raise ValueError("missing message_number")

                text = f"{item.get('message_text', '')} {item.get('reason', '')}".strip() or f"Message #{msg_no}"

                texts.append(text)
                ids.append(str(msg_no))
                metadatas.append(coerce_metadata(item))
            except Exception as e:
                skipped += 1
                print(f"  > SKIPPED item in batch. Reason: {e}")

        if not ids:
            continue

        try:
            # Generate new embeddings
            vecs = model.encode(texts, convert_to_numpy=True).tolist()
            embeddings.extend(vecs)

            # Upsert the batch to ChromaDB
            live_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            ok += len(ids)
            print(f"  > Upserted {len(ids)} items (last id: {ids[-1]})")
        except Exception as e:
            skipped += len(ids)
            print(f"  > BATCH FAILED ({len(ids)} items). Reason: {e}")

    return (ok, skipped, total)
def migrate_pending(pending_collection):
    """Migrates data from pending_solutions.json to the pending ChromaDB collection."""
    ok, skipped, total = 0, 0, 0

    if not PENDING_KB_PATH.exists():
        print("SKIPPED: pending_solutions.json not found.")
        return (ok, skipped, total)

    with open(PENDING_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Found {total} PENDING items. Upserting in batches of {BATCH_SIZE}...")

    for batch in chunked(data, BATCH_SIZE):
        ids, metadatas = [], []
        for item in batch:
            try:
                # Ensure a review_id exists
                review_id = item.get("review_id") or f"review_{uuid4().hex[:8]}"
                ids.append(str(review_id))
                metadatas.append(coerce_metadata(item))
            except Exception as e:
                skipped += 1
                print(f"  > SKIPPED pending item in batch. Reason: {e}")

        if not ids:
            continue

        try:
            # Upsert batch of pending items (metadata only)
            pending_collection.upsert(ids=ids, metadatas=metadatas)
            ok += len(ids)
            print(f"  > Upserted {len(ids)} pending items (last id: {ids[-1]})")
        except Exception as e:
            skipped += len(ids)
            print(f"  > PENDING BATCH FAILED ({len(ids)} items). Reason: {e}")

    return (ok, skipped, total)
# --------------------
# MAIN
# --------------------
def main():
    secrets = load_secrets()

    print("--- Connecting to ChromaDB Cloud ---")
    client = get_chroma_client(secrets)
    model = get_embedding_model()
    print(f"--- Connected. Tenant={secrets['CHROMA_TENANT']}, Database={secrets['CHROMA_DATABASE']} ---")

    # Define the embedding function that our collections will use
    embedding_func = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Get or create the collections, ensuring they use the correct embedding function
    live_collection = client.get_or_create_collection(name=LIVE_KB_COLLECTION, embedding_function=embedding_func)
    pending_collection = client.get_or_create_collection(name=PENDING_KB_COLLECTION, embedding_function=embedding_func)

    print("\n--- Migrating LIVE Knowledge Base ---")
    # Pass the collection objects directly to the functions
    ok_l, sk_l, tot_l = migrate_live(live_collection, model)
    print(f"LIVE: {ok_l}/{tot_l} upserted, {sk_l} skipped.")

    print("\n--- Migrating PENDING Solutions ---")
    ok_p, sk_p, tot_p = migrate_pending(pending_collection)
    print(f"PENDING: {ok_p}/{tot_p} upserted, {sk_p} skipped.")

    print("\n✅ All migrations finished.")



if __name__ == "__main__":
    sys.exit(main())
