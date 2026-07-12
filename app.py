import os
import json
import requests

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("AIPIPE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    invoice_text: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/extract")
def extract(data: InvoiceRequest):

    prompt = f"""
Extract the invoice information.

Return ONLY valid JSON.

Schema:

{{
  "invoice_no": null,
  "date": null,
  "vendor": null,
  "amount": null,
  "tax": null,
  "currency": null
}}

Rules:

- invoice_no = invoice/reference number
- date must always be YYYY-MM-DD
- vendor = seller/company
- amount = subtotal before tax
- tax = tax amount only
- currency = INR/USD/EUR etc.
- If missing return null.
- No explanation.
- No markdown.

Invoice:

{data.invoice_text}
"""

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an invoice extraction API. Always return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        },
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    text = result["choices"][0]["message"]["content"]

    data = json.loads(text)

    return {
        "invoice_no": data.get("invoice_no"),
        "date": data.get("date"),
        "vendor": data.get("vendor"),
        "amount": data.get("amount"),
        "tax": data.get("tax"),
        "currency": data.get("currency")
    }
