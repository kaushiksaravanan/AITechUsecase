import streamlit as st
import requests
import base64
import time
from cryptography.fernet import Fernet



# Use the same secret key as the server
SECRET_KEY = b'sHlkdKj_0-ZJ56RQlQmn8WS1TghwK31ZJ-ZYNcUgXrs=' # Replace with the actual key used in server.py
cipher = Fernet(SECRET_KEY)

# Define ntfy topic
NTFY_TOPIC = "my-secure-topic"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"  # Fetch messages from this URL

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

st.title("Stealth")

if st.button("Check for Notifications"):
    st.write("Decrypting message...")
    time.sleep(1)
    st.write("Almost there...")
    time.sleep(1)
    message = fetch_latest_notification()
    st.write("### Latest Notification:")
    st.write(message)

st.write("## Chat")
user_input = st.text_input("Enter your message:")
if st.button("Send Message") and user_input:
    send_message(user_input)