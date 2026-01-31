from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

import httpx
from opencode_ai import Opencode


def send_message(message: str, timeout: int, agent: str | None = None) -> str:
    client = Opencode()
    config = client.config.get()
    model_ref = getattr(config, "model", None)
    if not model_ref:
        raise RuntimeError("Opencode model not configured")

    provider_id, model_id = model_ref.split("/", 1)
    session_resp = client._client.post(
        "/session",
        json={},
        timeout=httpx.Timeout(15.0),  # timeout for session creation
        headers={"Content-Type": "application/json"},
    )
    session_resp.raise_for_status()  # raises HTTPStatusError for bad session
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

    if agent is not None:
        payload["agent"] = agent

    try:
        response = client._client.post(
            f"/session/{session_id}/message",
            json=payload,
            timeout=httpx.Timeout(timeout),  # timeout for LLM doing stuff
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        abort_session(client, session_id)
        raise
    body = response.json()
    return extract_text(body)


def extract_text(data: dict[str, Any]) -> str:
    def extract_from_parts(parts: Any) -> str:
        if not isinstance(parts, list):
            return ""
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
        return "".join(texts)

    text = extract_from_parts(data.get("parts"))
    if text:
        return text

    message = data.get("message")
    if isinstance(message, dict):
        text = extract_from_parts(message.get("parts"))
        if text:
            return text

    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict):
                text = extract_from_parts(msg.get("parts"))
                if text:
                    return text

    content = data.get("content")
    if isinstance(content, str) and content:
        return content

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_message = choice.get("message") or choice.get("delta")
            if isinstance(choice_message, dict):
                content = choice_message.get("content")
                if isinstance(content, str) and content:
                    return content
            content = choice.get("content")
            if isinstance(content, str) and content:
                return content

    return ""


def abort_session(client: Opencode, session_id: str) -> None:
    """
    Aborts a running OpenCode session.

    :param client: OpenCode client instance to use for the request
    :param session_id: Identifier of the session to abort
    """
    try:
        response = client._client.post(
            f"/session/{session_id}/abort",
            json={},
            timeout=httpx.Timeout(10.0),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None