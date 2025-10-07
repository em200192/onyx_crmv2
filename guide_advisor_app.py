import streamlit as st
import json
import os
from pathlib import Path
from uuid import uuid4
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"

# --- NEW: Configuration for Multiple KBs ---
# A dictionary to map module names to their JSON file paths
KB_PATHS = {
    "GL": CACHE_DIR / "gl_guide_kb.json",
    "Sales": CACHE_DIR / "sales_guide_kb.json",
    "Purchases": CACHE_DIR / "purchases_guide_kb.json",
    "Inventory": CACHE_DIR / "inventory_guide_kb.json",
}

# A dictionary to map module names to their image folders
IMG_DIRS = {
    "GL": CACHE_DIR / "guide_images" / "gl",
    "Sales": CACHE_DIR / "guide_images" / "sales",
    "Purchases": CACHE_DIR / "guide_images" / "purchases",
    "Inventory": CACHE_DIR / "guide_images" / "inventory",
}

# Ensure all directories exist
for path in IMG_DIRS.values():
    os.makedirs(path, exist_ok=True)

# --- PAGE SETUP ---
st.set_page_config(page_title="Guide Advisor Panel", layout="wide")
st.title("✍️ Guide Content Advisor Panel")
st.info("Use this page to add new topics to the User Guide knowledge bases.")


# --- HELPER FUNCTIONS ---
@st.cache_resource
def get_embedding_model():
    """Loads the sentence transformer model."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# --- MODIFIED: This function now takes a path argument ---
def add_topic_to_guide_kb(new_topic: dict, target_kb_path: Path):
    """
    Loads a specific guide KB, generates an embedding, appends the new topic, and saves.
    """
    try:
        if target_kb_path.exists():
            with open(target_kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
        else:
            kb = []

        model = get_embedding_model()
        text_to_embed = f"Title: {new_topic['title']}\nCategory: {new_topic['category']}\nContent: {new_topic['body']}"
        embedding = model.encode(text_to_embed).tolist()
        new_topic["embedding"] = embedding

        kb.append(new_topic)

        with open(target_kb_path, "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"Failed to update Knowledge Base: {e}")
        return False


# --- MAIN UI ---

# --- NEW: Module selection dropdown ---
st.subheader("Step 1: Choose the Knowledge Base")
module_choice = st.selectbox(
    "Select the module to add a new topic to:",
    ["GL", "Sales", "Purchases", "Inventory"]
)

# Determine the correct paths based on the user's choice
target_kb_path = KB_PATHS[module_choice]
target_img_dir = IMG_DIRS[module_choice]
st.info(f"You are currently adding a topic to the **{module_choice}** knowledge base.")
st.markdown("---")

with st.form("new_topic_form", clear_on_submit=True):
    st.subheader("Step 2: Enter the New Topic Details")

    topic_title = st.text_input("Topic Title*", help="Example: `سند القبض / Receipt Voucher`")

    category = st.selectbox(
        "Category*",
        ["Inputs", "Operations", "Configuration", "Reports"],
        help="Choose the category that best fits this topic."
    )

    answer_body = st.text_area("Answer / Body*", height=300,
                               help="Provide the full explanation or step-by-step guide for this topic.")

    solution_image = st.file_uploader("Upload Solution Image (Optional)", type=["png", "jpg", "jpeg"])

    submitted = st.form_submit_button(f"Add Topic to {module_choice} Knowledge Base")

    if submitted:
        if not all([topic_title, category, answer_body]):
            st.error("Please fill in all required fields marked with *.")
        else:
            image_path_list = []

            if solution_image is not None:
                try:
                    safe_title = "".join(c for c in topic_title if c.isalnum() or c in " -_").rstrip()[:30]
                    file_extension = solution_image.name.split('.')[-1]
                    unique_filename = f"{safe_title}_{uuid4().hex[:6]}.{file_extension}"

                    # --- MODIFIED: Use the dynamic image directory ---
                    image_path = target_img_dir / unique_filename

                    with open(image_path, "wb") as f:
                        f.write(solution_image.getbuffer())

                    image_path_list.append(str(image_path))
                    st.info(f"Image saved to {image_path}")
                except Exception as e:
                    st.error(f"Error saving image: {e}")

            new_topic_entry = {
                "title": topic_title.strip(),
                "category": category,
                "body": answer_body.strip(),
                "images": image_path_list,
                # Add the source module for easier tracking later
                "source_module": module_choice
            }

            with st.spinner(f"Saving to {module_choice} knowledge base..."):
                # --- MODIFIED: Pass the dynamic path to the function ---
                success = add_topic_to_guide_kb(new_topic_entry, target_kb_path)

            if success:
                st.success(f"Successfully added topic '{topic_title}' to the {module_choice} KB!")
                st.balloons()
            else:
                st.error("There was an error saving the new topic.")