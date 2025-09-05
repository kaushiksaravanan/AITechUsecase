import streamlit as st
import requests
import base64
import time
from cryptography.fernet import Fernet
from streamlit_autorefresh import st_autorefresh

# Use the same secret key as the server
SECRET_KEY = b'sHlkdKj_0-ZJ56RQlQmn8WS1TghwK31ZJ-ZYNcUgXrs='  # Replace with your actual key
cipher = Fernet(SECRET_KEY)

# Define ntfy topic
NTFY_TOPIC = "my-hcl"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"  # URL for fetching and posting messages

def fetch_latest_notification():
    try:
        response = requests.get(NTFY_URL)
        if response.status_code == 200:
            encrypted_message = base64.urlsafe_b64decode(response.text.encode())
            decrypted_message = cipher.decrypt(encrypted_message).decode()
            return decrypted_message
        else:
            return "No new notifications."
    except Exception as e:
        return f"Error: {e}"

def send_message(message: str):
    encrypted_message = cipher.encrypt(message.encode())
    encoded_message = base64.urlsafe_b64encode(encrypted_message).decode()
    requests.post(NTFY_URL, data=encoded_message.encode())
    st.success("Message sent!")

# Initialize session state for notification history if it doesn't exist
if "notification_history" not in st.session_state:
    st.session_state.notification_history = []

st.title("HCL Notif Client")

# Manual check for notifications
if st.button("Check for Notifications"):
    message = fetch_latest_notification()
    st.write("### Latest Notification:")
    st.write(message)
    # Append to history if it's a new message
    if not st.session_state.notification_history or st.session_state.notification_history[-1] != message:
        st.session_state.notification_history.append(message)

# Auto-refresh notifications using a checkbox toggle
if st.checkbox("Enable Auto-Refresh", key="auto_refresh_checkbox"):
    # This component forces a rerun every 2000 ms (2 seconds)
    st_autorefresh(interval=2000, limit=100, key="auto_refresh")
    new_message = fetch_latest_notification()
    # Only append if it's different from the last message in history
    if not st.session_state.notification_history or st.session_state.notification_history[-1] != new_message:
        st.session_state.notification_history.append(new_message)

# Display the history of notifications
st.subheader("Notification History")
for msg in st.session_state.notification_history:
    st.write(f"{msg} | Time: {time.strftime('%H:%M:%S')}")

st.write("## Chat")
user_input = st.text_input("Enter your message:")
if st.button("Send Message") and user_input:
    send_message(user_input)
