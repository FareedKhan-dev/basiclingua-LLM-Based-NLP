# FastAPI Wrapper

This is a simple FastAPI wrapper example

# Install

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app
```

# dotenv

You'll need a ```.env``` file.  It needs an entry for the Gemini API key whether you 
want to hardcode it or not; I set it to *redacted* - this means that clients
of the API will need to provide their Gemini key as a header parameter
to use the API.

# Dockerfile

Dockerfile example provided
