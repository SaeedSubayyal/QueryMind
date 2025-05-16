

import requests
import json

# Replace with your actual API key
API_KEY = "AIzaSyCYp6A_GYCYvnIEANyJw7GqOCZbqEhD1SM"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Request payload
payload = {
    "contents": [
        {
            "parts": [
                {"text": "Explain how AI works"}
            ]
        }
    ]
}

# Make POST request
headers = {
    "Content-Type": "application/json"
}

response = requests.post(URL, headers=headers, data=json.dumps(payload))

# Parse and print response
if response.status_code == 200:
    data = response.json()
    try:
        generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
        print("Gemini response:", generated_text)
    except (KeyError, IndexError):
        print("Unexpected response structure:", data)
else:
    print("Error:", response.status_code, response.text)
