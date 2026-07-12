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
    try:

        prompt = f"""
Extract the following invoice fields.

Return ONLY a valid JSON object.

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

- date must always be YYYY-MM-DD
- amount = subtotal before tax
- tax = tax amount only
- currency = INR/USD/EUR/etc.
- If a field is missing, return null.
- No markdown.
- No explanation.

Invoice:

{data.invoice_text}
"""

        response = requests.post(
            "https://aipipe.org/geminiv1beta/models/gemini-1.5-flash:generateContent",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        text = result["candidates"][0]["content"]["parts"][0]["text"]

        # Remove markdown if Gemini returns ```json ... ```
        text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)

        return {
            "invoice_no": data.get("invoice_no"),
            "date": data.get("date"),
            "vendor": data.get("vendor"),
            "amount": data.get("amount"),
            "tax": data.get("tax"),
            "currency": data.get("currency"),
        }

    except Exception as e:
        return {
            "error": str(e)
        }
