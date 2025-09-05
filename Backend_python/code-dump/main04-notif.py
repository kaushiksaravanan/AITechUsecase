import requests
import base64
import time
from cryptography.fernet import Fernet

# Generate or use a predefined encryption key
SECRET_KEY = Fernet.generate_key()
cipher = Fernet(SECRET_KEY)

# Define ntfy topic (channel)
NTFY_TOPIC = "my-secure-topic"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

# Threshold value
THRESHOLD = 10

def send_encrypted_notification(message: str):
    encrypted_message = cipher.encrypt(message.encode())
    encoded_message = base64.urlsafe_b64encode(encrypted_message).decode()
    
    print("Encrypting message...")
    time.sleep(1)
    print("Sending encrypted notification...")
    time.sleep(1)
    requests.post(NTFY_URL, data=encoded_message.encode())
    print("Encrypted notification sent!")

# Simulating a threshold check
value = 12  # Example value exceeding the threshold
if value > THRESHOLD:
    send_encrypted_notification(f"Alert! Value {value} exceeded the threshold.")
