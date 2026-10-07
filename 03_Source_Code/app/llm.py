"""Provider-agnostic LLM client.

Supports OpenAI, Azure OpenAI, Google Gemini and Anthropic Claude via environment variables.
If no key is configured the client reports `available == False` and every agent falls back to its
deterministic implementation, so the service always works (and tests run offline).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

from app.config import get_settings

log = logging.getLogger("netops.llm")


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0


class LLMClient:
    def __init__(self, provider: str | None = None):
        s = get_settings()
        self.settings = s
        self.provider = self._resolve(provider or s.llm_provider)
        self.model = {"openai": s.openai_model, "azure": os.getenv("AZURE_OPENAI_DEPLOYMENT", s.openai_model),
                      "gemini": s.gemini_model, "anthropic": s.anthropic_model}.get(self.provider, "none")
        self.stats = LLMStats()
        self._client = None

    @staticmethod
    def _resolve(p: str) -> str:
        p = (p or "auto").lower()
        if p != "auto":
            return p
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "none"

    @property
    def available(self) -> bool:
        return self.provider != "none"

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "available": self.available,
                "calls": self.stats.calls, "failures": self.stats.failures,
                "avg_latency_ms": round(self.stats.total_latency_ms / self.stats.calls, 1) if self.stats.calls else 0}

    # ------------------------------------------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        t = self.settings.llm_timeout_s
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(timeout=t)
        elif self.provider == "azure":
            from openai import AzureOpenAI
            self._client = AzureOpenAI(api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"), timeout=t)
        elif self.provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(timeout=t)
        return self._client

    def generate(self, system: str, user: str, json_mode: bool = False, max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        if not self.available:
            raise RuntimeError("No LLM provider configured")
        max_tokens = max_tokens or self.settings.llm_max_tokens
        temperature = self.settings.llm_temperature if temperature is None else temperature
        t0 = time.perf_counter()
        self.stats.calls += 1
        try:
            client = self._get_client()
            if self.provider in ("openai", "azure"):
                kw = {"response_format": {"type": "json_object"}} if json_mode else {}
                r = client.chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
                out = r.choices[0].message.content or ""
            elif self.provider == "gemini":
                from google.genai import types
                cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                  max_output_tokens=max_tokens,
                                                  response_mime_type="application/json" if json_mode else None)
                out = client.models.generate_content(model=self.model, contents=user, config=cfg).text or ""
            elif self.provider == "anthropic":
                sys_msg = system + ("\nRespond with a single valid JSON object only." if json_mode else "")
                r = client.messages.create(model=self.model, max_tokens=max_tokens,
                                           system=sys_msg, messages=[{"role": "user", "content": user}])
                out = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            else:
                raise RuntimeError(f"Unknown provider {self.provider}")
            return out
        except Exception:
            self.stats.failures += 1
            log.exception("LLM call failed (%s/%s)", self.provider, self.model)
            raise
        finally:
            self.stats.total_latency_ms += (time.perf_counter() - t0) * 1000

    def generate_json(self, system: str, user: str, **kw) -> dict:
        return parse_json(self.generate(system, user, json_mode=True, **kw))


def _sanitize_llm_text(raw: str) -> str:
    """Apply best-effort fixes to common LLM JSON defects *before* parsing."""
    s = raw

    # Strip JS-style comments (// … and /* … */) that appear outside strings
    s = re.sub(r'(?<!["\w])//[^\n]*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)

    # Replace literal (unescaped) newlines / tabs inside double-quoted strings
    # with their escaped equivalents so json.loads won't choke.
    def _escape_control(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return '"' + inner + '"'

    s = re.sub(r'"((?:[^"\\]|\\.)*?)"', _escape_control, s, flags=re.S)

    # Trailing commas  →  remove
    s = re.sub(r',\s*([\]\}])', r'\1', s)

    return s


def _single_to_double_quotes(raw: str) -> str:
    """Convert a JSON-like string that uses single quotes to double quotes.

    Handles apostrophes inside values (e.g. ``it's``) by only converting quotes
    that appear to act as JSON delimiters.
    """
    out: list[str] = []
    i = 0
    n = len(raw)
    in_str = False
    quote_char: str | None = None

    while i < n:
        ch = raw[i]

        # Inside a string
        if in_str:
            if ch == '\\' and i + 1 < n:
                out.append(ch)
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == quote_char:
                out.append('"')
                in_str = False
                quote_char = None
                i += 1
                continue
            # If the string was opened with single-quote, escape any inner double-quote
            if ch == '"' and quote_char == "'":
                out.append('\\"')
                i += 1
                continue
            out.append(ch)
            i += 1
            continue

        # Outside a string
        if ch in ('"', "'"):
            in_str = True
            quote_char = ch
            out.append('"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return ''.join(out)


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object in LLM output")
    raw = text[start:]

    # ---------- pipeline of increasingly aggressive repair attempts ----------

    # 1. Try standard json.loads with strict=False
    try:
        return json.loads(raw, strict=False)
    except Exception:
        pass

    # 2. Try trimming to the last closing brace (trailing prose after the JSON)
    end = raw.rfind("}")
    if end != -1:
        candidate = raw[:end + 1]
        try:
            return json.loads(candidate, strict=False)
        except Exception:
            pass

    # 3. Sanitise (comments, trailing commas, control chars inside strings)
    sanitised = _sanitize_llm_text(raw[:end + 1] if end != -1 else raw)
    try:
        return json.loads(sanitised, strict=False)
    except Exception:
        pass

    # 4. Single-quote → double-quote conversion (common with some models)
    sq_fixed = _single_to_double_quotes(sanitised)
    try:
        return json.loads(sq_fixed, strict=False)
    except Exception:
        pass

    # 5. Auto-repair unclosed quotes and brackets (e.g. if truncated by max_tokens)
    stack: list[str] = []
    in_str = False
    escape = False
    repaired: list[str] = []
    for ch in sq_fixed:
        if escape:
            repaired.append(ch)
            escape = False
            continue
        if ch == "\\":
            repaired.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            repaired.append(ch)
            continue
        if not in_str:
            if ch in ('{', '['):
                stack.append("}" if ch == "{" else "]")
            elif ch in ('}', ']'):
                if stack and stack[-1] == ch:
                    stack.pop()
        repaired.append(ch)

    if in_str:
        repaired.append('"')

    while stack:
        repaired.append(stack.pop())

    repaired_str = "".join(repaired)
    repaired_str = re.sub(r",\s*([\]\}])", r"\1", repaired_str)

    try:
        return json.loads(repaired_str, strict=False)
    except Exception:
        pass

    # 6. Last fallback: strip dangling incomplete key-value at the end
    base = raw[:raw.rfind("}") + 1] if "}" in raw else raw
    cleaned = re.sub(r',\s*"[^"]*":?\s*[^,}\]]*$', "", base)
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
    return json.loads(cleaned, strict=False)


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
