from __future__ import annotations

import os
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx
from opencode_ai import Opencode

DEFAULT_PROMPT = (
    "You are ChemMatData's research assistant. Given a publication URL, locate where the "
    "associated dataset can be downloaded (supplementary info, figshare/Zenodo/GitHub, "
    "institutional repository, etc.). Return the canonical dataset link and list the files that "
    "contain the usable data (CSV, JSON, HDF5, etc.). If no link is found, reply with 'DATA LINK "
    "NOT FOUND' and briefly describe what you checked. Reply exactly in this format:\n"
    "- Dataset link: <URL or DATA LINK NOT FOUND>\n"
    "- Key files:\n"
    "  - <file name>: <why it matters>\n"
    "- Notes: <how you located the link or why it is missing>"
)


def send_message(message: str) -> str:
    """
    Sends ``message`` to the configured Opencode provider and returns the reply text.

    :param message: The user prompt to forward to the model.

    :returns: The assistant reply text supplied by Opencode.
    """
    client = Opencode()
    config = client.config.get()
    model_ref = getattr(config, "model", None)
    if not model_ref:
        raise RuntimeError(
            "Opencode model is not configured. Run `opencode config set model provider/model` first."
        )
    provider_id, model_id = model_ref.split("/", 1)
    session_resp = client._client.post(  # type: ignore[attr-defined]  # noqa: SLF001
        "/session",
        json={},
        timeout=httpx.Timeout(30.0),
        headers={"Content-Type": "application/json"},
    )
    session_resp.raise_for_status()
    session_id = session_resp.json()["id"]

    payload: Dict[str, Any] = {
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
        response = client._client.post(  # type: ignore[attr-defined]  # noqa: SLF001
            f"/session/{session_id}/message",
            json=payload,
            timeout=httpx.Timeout(60.0),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - best effort debugging aid
        resp = exc.response
        raise RuntimeError(
            f"POST /session/{session_id}/message failed "
            f"with {resp.status_code}: {resp.text}\nPayload: {payload}"
        ) from exc
    except httpx.HTTPError as exc:  # pragma: no cover - best effort debugging aid
        raise RuntimeError(
            f"Failed to post message due to network error: {payload}"
        ) from exc
    body: Dict[str, Any]
    try:
        body = response.json()
    except ValueError:
        body = {}

    text = _extract_text(body)
    if text:
        return text

    history = client._client.get(  # type: ignore[attr-defined]  # noqa: SLF001
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
        text = _extract_text({"parts": entry.get("parts", [])})
        if text:
            return text

    return ""


def send_message_with_prompt(link: str) -> str:
    """
    Sends the configured research prompt with the given ``link`` appended and returns the reply.

    :param link: The publication URL that should be appended to the base prompt.

    :returns: The assistant reply text supplied by Opencode.
    """
    prompt = _build_prompt(link)
    return send_message(prompt)


def _extract_text(data: Dict[str, Any]) -> Optional[str]:
    """
    Returns the first text part contained in ``data`` or ``None``.
    """
    for part in data.get("parts", []):
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text")
    return None


def _build_prompt(link: str) -> str:
    """
    Builds the final prompt by combining the OPENCODE_PROMPT value with ``link``.
    """
    base_prompt = os.environ.get("OPENCODE_PROMPT", DEFAULT_PROMPT).strip()
    return f"{base_prompt}\n\nPublication: {link}".strip()
