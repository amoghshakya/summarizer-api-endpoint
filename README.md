# Bare-bones API endpoint for summarizer

This is a simple FastAPI application that provides an endpoint for summarizing
text. It only accepts POST requests with a JSON payload containing the text to
be summarized.

```jsonc
// Payload example
{
    "text": "Your text here"
}
```

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Transformers
- PyTorch
- gTTS
- pydantic

## Installation

```sh
pip install fastapi uvicorn transformers torch gtts pydantic
```

You can install all the requirements using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

Or use `environment.yml` if you are using conda.

```sh
conda create --name <env> --file <this file>
```

## Configuration

The model's path can be set within `app/config.py` file.

## Running the Application

```sh
uvicorn app.main:app --reload
```

> Omit the `--reload` flag for production.

## Send a POST request

```py
import requests

API_URL = "http://127.0.0.1:8000" # Change this if it differs

input_text = """
Some text to be summarized
"""

payload = {
    "text": input_text
}

response = requests.post(f"{API_URL}/summarize", json=payload)

if response.status_code == 200:
    data = response.json()
    print("Summary: ", data.get("summary", "No summary found"))
    if data["audio_url"]:
        print("Audio URL: ", data["audio_url"])
else:
    print("Error:", response.status_code)
    print("Response:", response.text)
```

or use `curl`

```sh
$ curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '{"text": "Your text here"}'
```
