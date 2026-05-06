"""
Block until the RM server in results/mvp/rm_url.txt is reachable and serving.

Tries up to ``--timeout-s`` seconds; exits 0 on first success, 1 on timeout.

Also fires one test query so we know the chat endpoint actually works (not just
that the model list endpoint returns).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_FILE = os.path.join(PROJECT_DIR, "results", "mvp", "rm_url.txt")


def http_post_json(url, payload, timeout=30):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--poll-s", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(URL_FILE):
        print(f"[healthcheck] URL file not found: {URL_FILE}", flush=True)
        # Keep trying — file may appear later.

    t0 = time.time()
    base_url = None
    while time.time() - t0 < args.timeout_s:
        if os.path.exists(URL_FILE):
            base_url = open(URL_FILE).read().strip()
        if base_url:
            try:
                m = http_get_json(f"{base_url}/models", timeout=5)
                if m.get("data"):
                    print(f"[healthcheck] /v1/models OK at {base_url}", flush=True)
                    break
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
                pass
        time.sleep(args.poll_s)
        elapsed = int(time.time() - t0)
        print(f"[healthcheck] waiting... {elapsed}s elapsed", flush=True)
    else:
        print(f"[healthcheck] TIMEOUT after {args.timeout_s}s", flush=True)
        return 1

    # Test chat completion
    try:
        r = http_post_json(f"{base_url}/chat/completions", {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "Reply with just the digit 7."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "seed": 42,
        }, timeout=60)
        out = r["choices"][0]["message"]["content"]
        print(f"[healthcheck] chat OK; sample reply: {out!r}", flush=True)
    except Exception as e:
        print(f"[healthcheck] chat FAILED: {e}", flush=True)
        return 1

    print(f"[healthcheck] PASS  url={base_url}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
