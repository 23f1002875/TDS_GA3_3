import os
import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
async def extract(data: InvoiceRequest):

    prompt = f"""
Extract the following fields from this invoice.

Return ONLY valid JSON.

Schema:

{{
  "invoice_no": string or null,
  "date": string (YYYY-MM-DD) or null,
  "vendor": string or null,
  "amount": number or null,
  "tax": number or null,
  "currency": string or null
}}

Rules:

- invoice_no = invoice/reference number
- vendor = seller/company name
- amount = subtotal BEFORE tax
- tax = only tax amount
- date must always be YYYY-MM-DD
- currency should be INR, USD, EUR etc.
- If missing return null.
- No markdown.
- No explanation.

Invoice:

{data.invoice_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    result = json.loads(response.text)

    output = {
        "invoice_no": result.get("invoice_no"),
        "date": result.get("date"),
        "vendor": result.get("vendor"),
        "amount": result.get("amount"),
        "tax": result.get("tax"),
        "currency": result.get("currency"),
    }

    return output
