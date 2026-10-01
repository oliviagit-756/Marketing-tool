"""
Phase 2: message generation with the Groq API (OpenAI-compatible).
- generate_templates():  one LLM call per segment -> {segment_name: {"subject", "body", "source"}}
- render_message():      fill {first_name} / {product} for one customer

Setup (environment variables):
  GROQ_API_KEY   from https://console.groq.com/keys
  GROQ_MODEL     a chat model ID that your Groq free plan lists (check Groq's models page)
"""
import json
import logging
import os
import re

import requests

try:  # local runs: read GROQ_API_KEY / GROQ_MODEL from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # in Lambda the values come from the function's environment variables
    pass

from model import SEGMENTS

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT_SECONDS = 30

MAX_SUBJECT_CHARS = 80
MAX_BODY_CHARS = 700

SYSTEM_PROMPT = (
    "You are an email copywriter for an online retailer. "
    "Write short, warm, professional marketing emails. "
    "Never invent discount amounts, percentages, prices, promo codes, or deadlines. "
    "Refer to offers only in general terms such as 'a special offer' or 'an exclusive reward'. "
    "Do not write a sign-off or signature; end with the call to action. "
    "Never use square brackets or placeholders other than {first_name} and {product}. "
    "Do not make factual claims about stock, restocks, new products, new styles, or service improvements. "
    "Do not mention links, buttons, or clicking. "
    "Stay within the campaign goal: only imply the customer has been away if the goal is about winning them back. "
    "Use correct spacing and punctuation. "
    "Respond with JSON only, no markdown fences, no extra text. "
    " only plain ASCII punctuation: straight apostrophes and quotes, no curly quotes."
)


def _sign_off():
    brand = os.environ.get("BRAND_NAME", "The Customer Care Team")
    return f"\n\nWarm regards,\n{brand}"


def _build_prompt(segment):
    return (
        f"Customer segment: {segment['name']}\n"
        f"Campaign goal: {segment['goal']}\n"
        f"Tone: {segment['tone']}\n\n"
        "Write ONE email for this segment. Use these exact placeholders where they fit naturally: "
        "{first_name} for the customer's first name and {product} for the product they buy most.\n"
        f"Subject: max {MAX_SUBJECT_CHARS} characters. Body: max 90 words, plain text, "
        "end with a short call to action.\n\n"
        'Return JSON: {"subject": "...", "body": "..."}'
    )


def _fallback_template(segment):
    return {
        "subject": "A message for you, {first_name}",
        "body": "Hi {first_name},\n\n" + segment["fallback"] + _sign_off(),
        "source": "fallback",
    }


def _parse_json(text):
    """Strip code fences / stray text and parse the first JSON object."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model reply")
    return json.loads(match.group(0))


def _validate(data, winback_ok=False):
    """Guardrails: right keys, length limits, placeholder present, no made-up claims."""
    subject, body = data["subject"].strip(), data["body"].strip()
    text = subject + " " + body
    if not subject or not body:
        raise ValueError("empty subject or body")
    if len(subject) > MAX_SUBJECT_CHARS or len(body) > MAX_BODY_CHARS:
        raise ValueError("too long")
    if "{product}" not in body:
        raise ValueError("missing {product} placeholder")
    if re.search(r"[%$£€]|\bcode\b", text, flags=re.IGNORECASE):
        raise ValueError("contains a made-up discount/price/code")
    if re.search(r"[\[\]]", text):
        raise ValueError("contains a bracket placeholder")
    if re.search(r"https?://|www\.|click|\blink\b|button", text, flags=re.IGNORECASE):
        raise ValueError("mentions a link or button")
    if re.search(r"in stock|restock|new (?:styles?|arrivals?|collections?|selections?|products?|items?)"
                 r"|busy adding|what.s new|improv\w+ our (?:service|store)", text, flags=re.IGNORECASE):
        raise ValueError("makes an invented claim (stock / new products / service)")
    if not winback_ok and re.search(
            r"\bmiss(?:ed)?\b|been a while|welcome back|haven.t seen",
            text, flags=re.IGNORECASE):
        raise ValueError("implies the customer has been away")
    return {"subject": subject, "body": body + _sign_off(), "source": "groq"}


MAX_ATTEMPTS = 3  # retries when a reply is cut off, unparsable, or fails a guardrail


def _call_groq(api_key, model_id, segment):
    """One API call; returns the model's text. Raises if the reply was cut off."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(segment)},
        ],
        "temperature": 0.5,
        # reasoning models spend part of this budget "thinking" before they answer
        "max_completion_tokens": 1500,
    }
    if "gpt-oss" in model_id:
        payload["reasoning_effort"] = "low"  # less thinking -> the answer is not cut off
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("reply was cut off (token limit reached)")
    return choice["message"]["content"]


def generate_segment_template(segment):
    """Call Groq for one segment. Falls back to the static template on any failure."""
    api_key = os.environ.get("GROQ_API_KEY")
    model_id = os.environ.get("GROQ_MODEL")
    if not api_key or not model_id:
        logger.warning("GROQ_API_KEY / GROQ_MODEL not set; using fallback for %s", segment["name"])
        return _fallback_template(segment)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            text = _call_groq(api_key, model_id, segment)
            winback_ok = segment["name"] == SEGMENTS[-1]["name"]  # only the last segment may say 'we miss you'
            return _validate(_parse_json(text), winback_ok=winback_ok)
        except requests.HTTPError as exc:  # 400/401/429 etc: retrying will not help
            logger.warning("Groq HTTP error for %s (%s)", segment["name"], exc)
            break
        except Exception as exc:  # cut off, bad JSON, failed guardrail, timeout -> try again once
            logger.warning("Groq attempt %d/%d failed for %s (%s)",
                           attempt, MAX_ATTEMPTS, segment["name"], exc)
    logger.warning("Using fallback template for %s", segment["name"])
    return _fallback_template(segment)


def generate_templates():
    """One template per segment (4 LLM calls total)."""
    return {s["name"]: generate_segment_template(s) for s in SEGMENTS}


def render_message(template, first_name, product):
    """Fill placeholders for one customer. Uses replace(), so stray braces can't crash it."""
    fill = lambda t: t.replace("{first_name}", first_name).replace("{product}", product)
    return fill(template["subject"]), fill(template["body"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name, tpl in generate_templates().items():
        subject, body = render_message(tpl, "Alex", "Vintage Teapot")
        print(f"\n=== {name}  [{tpl['source']}] ===\nSubject: {subject}\n\n{body}")