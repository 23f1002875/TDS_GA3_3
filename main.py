import json
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import config

app = FastAPI()

# CORS wide open — grader calls from a Cloudflare Worker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

HEAD = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}

REQUIRED_KEYS = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]


async def chat(messages, model=None, max_tokens=800, retries=3):
    """Call AIPipe's OpenAI-compatible chat endpoint with basic retry on transient errors."""
    body = {
        "model": model or config.TEXT_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            try:
                r = await c.post(f"{config.AIPIPE_BASE}/chat/completions", headers=HEAD, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = str(e)
    raise RuntimeError(f"chat failed after {retries} attempts: {last_err}")


def parse_json(s):
    """Strip markdown fences if present and parse JSON, falling back to regex extraction."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def to_number(v):
    """Coerce a value (possibly a string with commas/currency symbols) to a float, or None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    s = re.sub(r"[,\s₹$€£]", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def clean_str(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    return s


last_debug_info = {}


@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}


@app.get("/debug")
async def get_debug():
    """After a failed grader Check, open https://<your-url>.onrender.com/debug
    to see exactly what was sent to the model and what came back."""
    return last_debug_info


@app.post("/extract")
async def extract(request: Request):
    global last_debug_info
    body = await request.json()
    text = body.get("invoice_text", "")

    last_debug_info = {"invoice_text": text}

    # Safe fallback object — ALWAYS has all 6 keys, even if everything below fails.
    result = {k: None for k in REQUIRED_KEYS}

    prompt = (
        "Extract these fields from the invoice text and return JSON with "
        "EXACTLY these keys: invoice_no, date, vendor, amount, tax, currency.\n"
        "- invoice_no: the invoice/reference number exactly as written "
        "(e.g. 'NS/2026/778', 'INV-2026-0041')\n"
        "- date: the invoice/issue date, converted to ISO format YYYY-MM-DD "
        "(e.g. '15 March 2026' -> '2026-03-15')\n"
        "- vendor: the company/person ISSUING the invoice (not the bill-to/client)\n"
        "- amount: the SUBTOTAL before tax, as a plain number (no commas, no "
        "currency symbols — just digits and an optional decimal point). Do NOT "
        "confuse this with the grand total.\n"
        "- tax: the tax amount only (GST/IGST/VAT/etc.), as a plain number\n"
        "- currency: the ISO currency code (INR, USD, EUR...). If the text uses "
        "'Rs.' or '₹', use 'INR'. If '$', use 'USD'. If not determinable, use null.\n"
        "- Use JSON null (not the string \"null\") for any field that is genuinely "
        "not present in the text.\n"
        "Return ONLY a JSON object with exactly these 6 keys, nothing else.\n\n"
        f"TEXT:\n{text}"
    )

    try:
        raw = await chat([{"role": "user", "content": prompt}], max_tokens=500)
        last_debug_info["raw_model_output"] = raw
        out = parse_json(raw)
        last_debug_info["parsed_json"] = out

        result["invoice_no"] = clean_str(out.get("invoice_no"))
        result["date"] = clean_str(out.get("date"))
        result["vendor"] = clean_str(out.get("vendor"))
        result["amount"] = to_number(out.get("amount"))
        result["tax"] = to_number(out.get("tax"))
        result["currency"] = clean_str(out.get("currency"))

    except Exception as e:
        last_debug_info["exception"] = str(e)
        # result already has all 6 keys defaulted to None — safe to return as-is

    last_debug_info["final_result"] = result

    # Guarantee: always return exactly these 6 keys, regardless of what happened above.
    return {k: result.get(k) for k in REQUIRED_KEYS}
