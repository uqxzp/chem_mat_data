import argparse
import sys

import httpx

PROMPT = """\
Sanity check for custom tool.

1) Call the tool `crawlfetch` with url="{url}" and maxChars=3000.
2) In your final answer, print EXACTLY this line: CRAWLFETCH_OK
3) Then include a short summary (1-3 sentences) of what you saw.

Do not use any other web tools.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url", default="http://127.0.0.1:54321", help="OpenCode server base URL"
    )
    p.add_argument(
        "--url", default="https://example.com", help="URL to fetch via crawlfetch"
    )
    p.add_argument(
        "--agent", default="build", help='Agent name, e.g. "build" or "plan"'
    )
    p.add_argument(
        "--user", default=None, help="HTTP basic auth username (if server is protected)"
    )
    p.add_argument(
        "--password",
        default=None,
        help="HTTP basic auth password (if server is protected)",
    )
    args = p.parse_args()

    auth = (args.user, args.password) if (args.user and args.password) else None

    with httpx.Client(base_url=args.base_url, timeout=180.0, auth=auth) as client:
        # 1) Create a new session
        r = client.post("/session", json={"title": "crawlfetch sanity check"})
        r.raise_for_status()
        session = r.json()
        session_id = (
            session.get("id") or session.get("sessionID") or session.get("session_id")
        )
        if not session_id:
            print("Couldn't find session id in response:", session, file=sys.stderr)
            return 2

        # 2) Send a message (waits for response)
        body = {
            "agent": args.agent,
            "parts": [{"type": "text", "text": PROMPT.format(url=args.url)}],
        }
        r = client.post(f"/session/{session_id}/message", json=body)
        r.raise_for_status()
        msg = r.json()

    # 3) Print all text parts from the response
    parts = msg.get("parts", [])
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
    output = "\n".join(t for t in texts if t)

    print(output or msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
