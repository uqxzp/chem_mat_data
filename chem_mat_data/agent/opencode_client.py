from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import httpx
from opencode_ai import Opencode

# opencode serve --port 54321
# cmmanage agent process_link https://www.nature.com/articles/sdata201422
# cmmanage agent generate_script --mcp-endpoint "file-download (tool: download file)"

# TODO: Which formats exactly do we want/should the prompt contain?
DEFAULT_PROMPT = """
You are a research assistant focused on chemistry and materials science datasets.
Given a publication URL, find direct download URLs for the dataset (supplementary information,
Zenodo/Figshare/OSF/GitHub releases, institutional repositories, lab-hosted archives, etc.).
Only keep links that directly download files or archives containing SMILES, SDF, XYZ, or CIF data.

Return a JSON object only:
{
  "download_links": ["https://direct-download-1", "..."],
  "notes": "short bullet points about where the links were found or why none were found"
}

If nothing is available, return an empty list for "download_links" and explain in "notes".
"""


def send_message(message: str) -> str:
    """
    Sends the message to the configured Opencode provider and returns the reply text
    """
    client = Opencode()
    config = client.config.get()
    model_ref = getattr(config, "model", None)
    if not model_ref:
        raise RuntimeError(
            "Opencode model not configured"
        )
    provider_id, model_id = model_ref.split("/", 1)
    session_resp = client._client.post(
        "/session",
        json={},
        timeout=httpx.Timeout(30.0),
        headers={"Content-Type": "application/json"},
    )
    session_resp.raise_for_status()
    session_id = session_resp.json()["id"]

    payload: dict[str, Any] = {
        "messageID": f"msg_{uuid4().hex}",
        "model": {
            "providerID": provider_id,
            "modelID": model_id,
        },
        "parts": [
            {
                "type": "text",
                "text": message,
            }
        ],
    }

    try:
        response = client._client.post(
            f"/session/{session_id}/message",
            json=payload,
            timeout=httpx.Timeout(60.0),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        raise RuntimeError(
            f"POST /session/{session_id}/message failed "
            f"with {resp.status_code}: {resp.text}\nPayload: {payload}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Failed to post message due to network error: {payload}"
        ) from exc
    body: dict[str, Any]
    try:
        body = response.json()
    except ValueError:
        body = {}

    text = extract_text(body)
    if text:
        return text

    history = client._client.get(
        f"/session/{session_id}/message",
        timeout=httpx.Timeout(30.0),
    )
    history.raise_for_status()
    try:
        messages = history.json()
    except ValueError as exc:
        raise RuntimeError("Failed to decode session history response") from exc

    for entry in reversed(messages):
        info = entry.get("info", {})
        if info.get("role") != "assistant":
            continue
        text = extract_text({"parts": entry.get("parts", [])})
        if text:
            return text

    return ""


def send_message_with_prompt(link: str) -> str:
    """
    Sends the configured research prompt with the given link appended and returns the reply
    """
    prompt = build_prompt(link)
    return send_message(prompt)


def extract_text(data: dict[str, Any]) -> str | None:
    """
    Returns the first text part contained in data or None
    """
    for part in data.get("parts", []):
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text")
    return None


def build_prompt(link: str) -> str:
    """
    Builds the final prompt by combining the OPENCODE_PROMPT value with the link
    """
    base_prompt = os.environ.get("OPENCODE_PROMPT", DEFAULT_PROMPT).strip()
    return f"{base_prompt}\n\nPublication: {link}".strip()
