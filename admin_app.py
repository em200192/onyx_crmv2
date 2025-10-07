import streamlit as st
import os
from pathlib import Path

# --- NEW: Import from our central database utility file ---
from db_utils import (
    load_pending_solutions_db,
    approve_solution_db,
    reject_solution_db
)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
SOLUTION_IMG_DIR = CACHE_DIR / "solution_images"  # Still needed to display images

# --- PAGE SETUP ---
st.set_page_config(page_title="Admin Panel", layout="wide")
st.title("🛡️ Admin Panel: Approve New Solutions")

# --- MAIN ADMIN UI with Session State ---

# Initialize state only if it's not already there.
if 'pending_solutions' not in st.session_state:
    st.session_state.pending_solutions = load_pending_solutions_db()

if not st.session_state.pending_solutions:
    st.success("✅ No solutions are currently pending review.")
else:
    st.info(f"There are **{len(st.session_state.pending_solutions)}** solutions awaiting your approval.")

    # Iterate over a copy of the list to avoid issues while modifying it
    for chroma_id, solution in st.session_state.pending_solutions[:]:
        st.markdown("---")
        review_id = solution.get("review_id")

        with st.container():

            st.subheader(f"Submission: Error `{solution.get('message_number')}`")

            st.markdown(f"**Message Text:** {solution.get('message_text')}")
            st.markdown(f"**Reason:** {solution.get('reason')}")
            st.markdown(f"**Solution Steps:**\n```\n{solution.get('solution')}\n```")

            if solution.get('note'):
                st.markdown(f"**Note:** {solution.get('note')}")

            if solution.get('image_path') and os.path.exists(solution.get('image_path')):
                st.image(solution.get('image_path'), caption="Solution Image")

            col1, col2 = st.columns(2)
            if col1.button("✅ Approve", key=f"approve_{chroma_id}", use_container_width=True):
                # --- MODIFIED: Use the database function ---
                approve_solution_db(solution)

                st.session_state.pending_solutions = [s for s in st.session_state.pending_solutions if s[0] != chroma_id]
                st.success(f"Solution approved and added to the live KB.")
                st.rerun()

            if col2.button("❌ Reject", key=f"reject_{chroma_id}", use_container_width=True):
                # --- MODIFIED: Use the database function ---
                reject_solution_db(chroma_id)

                st.session_state.pending_solutions = [s for s in st.session_state.pending_solutions if s[0] != chroma_id]
                st.warning(f"Solution rejected and removed from the queue.")
                st.rerun()