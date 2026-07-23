from __future__ import annotations

import base64
import re

INSTRUCTION_INJECTION_PATTERNS = (
    re.compile(r"\bi[\W_]*g[\W_]*n[\W_]*o[\W_]*r[\W_]*e\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bignore\s+system\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(the\s+)?(system|developer|user)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(the\s+)?system\s+prompt\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(in|a|an)\b", re.IGNORECASE),
    re.compile(r"\bexecute\s+these\s+instructions\b", re.IGNORECASE),
    re.compile(r"\b(fake\s+admin\s+message|admin\s+message)\s*:", re.IGNORECASE),
    re.compile(r"\b(publish|send)\s+(now|messages?|emails?|sms)\b", re.IGNORECASE),
    re.compile(r"\breveal\s+other\s+workspaces\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\s+data\b", re.IGNORECASE),
)
BASE64_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def contains_instruction_injection(text: str) -> bool:
    return any(
        pattern.search(text) is not None for pattern in INSTRUCTION_INJECTION_PATTERNS
    ) or any(
        _decoded_token_contains_instruction(token) for token in BASE64_TOKEN_PATTERN.findall(text)
    )


def normalize_source_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_retrieved_text(text: str) -> str:
    sanitized = text
    for pattern in INSTRUCTION_INJECTION_PATTERNS:
        sanitized = pattern.sub("[removed untrusted instruction]", sanitized)
    sanitized = BASE64_TOKEN_PATTERN.sub(_sanitize_base64_token, sanitized)
    return normalize_source_text(sanitized)


def _decoded_token_contains_instruction(token: str) -> bool:
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    return any(pattern.search(decoded) is not None for pattern in INSTRUCTION_INJECTION_PATTERNS)


def _sanitize_base64_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if _decoded_token_contains_instruction(token):
        return "[removed encoded untrusted instruction]"
    return token
