import streamlit as st
import os
from pathlib import Path
from uuid import uuid4
import re

# --- MODIFIED: Import all necessary db_utils functions ---
from db_utils import (
    load_escalations_gsheet,
    mark_escalation_as_done_gsheet,
    submit_solution_for_review_db
)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
SOLUTION_IMG_DIR = CACHE_DIR / "solution_images"
os.makedirs(SOLUTION_IMG_DIR, exist_ok=True)

# --- PAGE SETUP ---
st.set_page_config(page_title="Advisor Panel", layout="wide")
st.title("👨‍🏫 Advisor Panel: Knowledge Base Management")

# --- UI Layout with Tabs ---

# --- CORRECTED: Load pending escalations from Google Sheets ---
try:
    all_escalations = load_escalations_gsheet()
    pending_escalations = [row for row in all_escalations if row.get("status") == "pending"]
except Exception as e:
    st.error(f"Could not connect to Google Sheets. Please check secrets/sharing. Error: {e}")
    pending_escalations = []

pending_count = len(pending_escalations)

tab1, tab2 = st.tabs([f"📝 Pending Escalations ({pending_count})", "➕ Add New Error Manually"])

# --- Tab 1: Pending Escalations (Reads from Google Sheets, Writes to ChromaDB) ---
with tab1:
    if not pending_escalations:
        st.success("✅ No pending escalations to review.")
    else:
        st.info(f"You have {pending_count} pending escalations to resolve.")

        options_dict = {f"{row['ticket_id']}: {row['query']}": row for row in pending_escalations}
        selected_option_key = st.selectbox("Choose an escalation to resolve:", options_dict.keys())

        if selected_option_key:
            selected_row = options_dict[selected_option_key]

            st.subheader(f"Resolving Ticket: {selected_row['ticket_id']}")
            st.write(f"**Original User Query:** `{selected_row['query']}`")

            with st.form("resolve_form"):
                message_number = st.text_input("Message Number*",
                                               value=re.findall(r'\d+', selected_row['query'])[0] if re.findall(r'\d+',
                                                                                                                selected_row[
                                                                                                                    'query']) else "")
                message_text = st.text_area("Message Text*", value=selected_row['query'])
                location = st.text_input("Location / Screen")
                reason = st.text_area("Reason*")
                solution = st.text_area("Solution*")
                note = st.text_input("Important Note (Optional)")
                solution_image = st.file_uploader("Upload Solution Image (Optional)", type=["png", "jpg", "jpeg"],
                                                  key="resolve_uploader")

                submitted = st.form_submit_button("Submit Solution for Review")

                if submitted:
                    if not all([message_number, message_text, reason, solution]):
                        st.error("Please fill in all required fields marked with *.")
                    else:
                        image_path = None
                        if solution_image is not None:
                            file_extension = solution_image.name.split('.')[-1]
                            unique_filename = f"sol_{message_number.strip()}_{uuid4().hex[:6]}.{file_extension}"
                            image_path = str(SOLUTION_IMG_DIR / unique_filename)
                            with open(image_path, "wb") as f:
                                f.write(solution_image.getbuffer())

                        new_solution_entry = {
                            "ticket_id": selected_row['ticket_id'],
                            "message_number": message_number.strip(),
                            "message_text": message_text.strip(),
                            "location": location.strip(),
                            "reason": reason.strip(),
                            "solution": solution.strip(),
                            "note": note.strip(),
                            "image_path": image_path
                        }

                        # Step 1: Submit the completed solution to ChromaDB for the admin to approve.
                        submit_solution_for_review_db(new_solution_entry)

                        # Step 2: Mark the ticket as "done" in the Google Sheet.
                        mark_escalation_as_done_gsheet(selected_row['ticket_id'])

                        st.success(
                            "Successfully submitted solution for admin review! The ticket has been removed from your queue.")
                        st.rerun()

# --- Tab 2: Add New Error Manually (Writes directly to ChromaDB for admin approval) ---
with tab2:
    st.info("Use this form to proactively add a new error and solution to the knowledge base.")

    with st.form("add_new_form", clear_on_submit=True):
        st.subheader("New Error Details")

        message_number = st.text_input("Message Number*")
        message_text = st.text_area("Message Text*")
        location = st.text_input("Location / Screen")
        reason = st.text_area("Reason*")
        solution = st.text_area("Solution*")
        note = st.text_input("Important Note (Optional)")
        solution_image = st.file_uploader("Upload Solution Image (Optional)", type=["png", "jpg", "jpeg"],
                                          key="add_new_uploader")

        submitted = st.form_submit_button("Submit New Entry for Review")

        if submitted:
            if not all([message_number, message_text, reason, solution]):
                st.error("Please fill in all required fields marked with *.")
            else:
                image_path = None
                if solution_image is not None:
                    file_extension = solution_image.name.split('.')[-1]
                    unique_filename = f"sol_{message_number.strip()}_{uuid4().hex[:6]}.{file_extension}"
                    image_path = SOLUTION_IMG_DIR / unique_filename
                    with open(image_path, "wb") as f:
                        f.write(solution_image.getbuffer())
                    image_path = str(image_path)

                new_solution_entry = {
                    "ticket_id": None,  # No ticket ID for manual entries
                    "message_number": message_number.strip(),
                    "message_text": message_text.strip(),
                    "location": location.strip(),
                    "reason": reason.strip(),
                    "solution": solution.strip(),
                    "note": note.strip(),
                    "image_path": image_path
                }

                # --- MODIFIED: Submit to ChromaDB for review ---
                submit_solution_for_review_db(new_solution_entry)
                st.success(f"Successfully submitted error #{message_number.strip()} for admin review!")
                st.balloons()