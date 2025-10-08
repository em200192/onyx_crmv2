import streamlit as st
import os
from pathlib import Path
from uuid import uuid4

# Import all the necessary db_utils functions
from db_utils import (
    load_pending_solutions_db,
    approve_solution_db,
    reject_solution_db,
    submit_solution_for_review_db, load_conversation_history_from_gcs  # We'll reuse this for saving edits
)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
SOLUTION_IMG_DIR = CACHE_DIR / "solution_images"
os.makedirs(SOLUTION_IMG_DIR, exist_ok=True)

# --- PAGE SETUP ---
st.set_page_config(page_title="Admin Panel", layout="wide")
st.title("🛡️ Admin Panel: Approve New Solutions")

# --- Initialize session state ---
if 'editing_item_id' not in st.session_state:
    st.session_state.editing_item_id = None
if 'pending_solutions' not in st.session_state:
    st.session_state.pending_solutions = load_pending_solutions_db()

# --- MAIN ADMIN UI ---
if not st.session_state.pending_solutions:
    st.success("✅ No solutions are currently pending review.")
    st.session_state.editing_item_id = None  # Clear editing state if list is empty
else:
    st.info(f"There are **{len(st.session_state.pending_solutions)}** solutions awaiting your approval.")

    # Use a copy of the list for iteration if we're modifying it
    for chroma_id, solution in list(st.session_state.pending_solutions):  # Use list() to iterate over a copy
        st.markdown("---")

        # --- NEW LOGIC: Check if the current item is in "edit mode" ---
        if st.session_state.editing_item_id == chroma_id:
            # --- RENDER THE EDIT FORM ---
            with st.form(key=f"edit_form_{chroma_id}"):
                st.subheader(f"Editing Error: `{solution.get('message_number', 'N/A')}`")

                edited_message_number = st.text_input("Message Number*", value=solution.get("message_number", ""))
                edited_message_text = st.text_area("Message Text*", value=solution.get("message_text", ""))
                edited_location = st.text_input("Location / Screen", value=solution.get("location", ""))
                edited_reason = st.text_area("Reason*", value=solution.get("reason", ""))
                edited_solution = st.text_area("Solution*", value=solution.get("solution", ""))
                edited_note = st.text_input("Important Note", value=solution.get("note", ""))

                # --- NEW: Image Editing ---
                current_image_path = solution.get('image_path')
                if current_image_path:
                    st.image(current_image_path, caption="Attached Image", width=300)
                else:
                    st.write("No current image.")

                new_solution_image_file = st.file_uploader("Upload New Solution Image (Optional)",
                                                           type=["png", "jpg", "jpeg"],
                                                           key=f"edit_image_uploader_{chroma_id}")

                col1, col2 = st.columns(2)
                if col1.form_submit_button("💾 Save Changes", use_container_width=True):
                    # Validate required fields
                    if not all([edited_message_number, edited_message_text, edited_reason, edited_solution]):
                        st.error("Please fill in all required fields marked with *.")
                    else:
                        # Handle new image upload
                        image_path_to_save = current_image_path  # Default to existing image
                        if new_solution_image_file:
                            file_extension = new_solution_image_file.name.split('.')[-1]
                            # Use a new UUID to prevent caching issues if the same filename is used
                            unique_filename = f"sol_{edited_message_number.strip()}_{uuid4().hex[:6]}.{file_extension}"
                            image_path_to_save = str(SOLUTION_IMG_DIR / unique_filename)
                            with open(image_path_to_save, "wb") as f:
                                f.write(new_solution_image_file.getbuffer())
                            st.info(f"New image saved: {image_path_to_save}")

                        # Create an updated solution dictionary
                        updated_solution = solution.copy()
                        updated_solution.update({
                            "review_id": chroma_id,  # <--- CRITICAL FIX: Pass the review_id to update
                            "message_number": edited_message_number.strip(),
                            "message_text": edited_message_text.strip(),
                            "location": edited_location.strip(),
                            "reason": edited_reason.strip(),
                            "solution": edited_solution.strip(),
                            "note": edited_note.strip(),
                            "image_path": image_path_to_save
                        })

                        # Reuse our submit function, which uses upsert to update the record
                        submit_solution_for_review_db(updated_solution)

                        # Exit edit mode and refresh the data
                        st.session_state.editing_item_id = None
                        # Force reload pending solutions by deleting it from session state
                        del st.session_state.pending_solutions
                        st.success("Changes saved!")
                        st.rerun()

                if col2.form_submit_button("❌ Cancel", use_container_width=True):
                    # Just exit edit mode
                    st.session_state.editing_item_id = None
                    st.rerun()

        else:
            # --- RENDER THE NORMAL DISPLAY MODE ---
            with st.container():
                st.subheader(f"Submission: Error `{solution.get('message_number', 'N/A')}`")
                if solution.get('ticket_id'):
                    st.caption(f"From Escalation Ticket: `{solution.get('ticket_id')}`")

                # Display chat history expander
                ticket_id = solution.get("ticket_id")
                if ticket_id:
                    history = load_conversation_history_from_gcs(ticket_id)
                    if history:
                        with st.expander("View Full Conversation History"):
                            for msg in history:
                                role = msg.get("role", "unknown").capitalize()
                                content = msg.get("content", "")
                                if role == "User":
                                    st.markdown(f"👤 **{role}:** {content}")
                                else:
                                    st.markdown(f"🤖 **{role}:** {content}")

                st.markdown(f"**Message Text:** {solution.get('message_text', 'N/A')}")
                st.markdown(f"**Location:** {solution.get('location', 'N/A')}")
                st.markdown(f"**Reason:** {solution.get('reason', 'N/A')}")
                st.markdown(f"**Solution:** {solution.get('solution', 'N/A')}")
                st.markdown(f"**Note:** {solution.get('note', 'N/A')}")

                current_image_path = solution.get('image_path')
                if current_image_path:
                    st.image(current_image_path, caption="Attached Image", width=300)
                elif current_image_path:
                    st.warning(f"Image not found at: {current_image_path}")

                col1, col2, col3 = st.columns(3)
                if col1.button("✅ Approve", key=f"approve_{chroma_id}", use_container_width=True):
                    approve_solution_db(solution)
                    del st.session_state.pending_solutions  # Force reload
                    st.success(f"Solution approved.")
                    st.rerun()

                if col2.button("❌ Reject", key=f"reject_{chroma_id}", use_container_width=True):
                    reject_solution_db(chroma_id)
                    del st.session_state.pending_solutions  # Force reload
                    st.warning(f"Solution rejected.")
                    st.rerun()

                if col3.button("✏️ Edit", key=f"edit_{chroma_id}", use_container_width=True):
                    # --- SET THE APP TO "EDIT MODE" FOR THIS ITEM ---
                    st.session_state.editing_item_id = chroma_id
                    st.rerun()
