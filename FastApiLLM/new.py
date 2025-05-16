import requests

# Your OpenAI API key
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

# The prompt you want to send to the model
data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you help me with Python programming?"}
    ]
}

# Send a POST request to the OpenAI API
response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)

# Print the status code and the response
print(response.status_code)
print(response.json())
