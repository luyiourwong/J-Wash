import asyncio
import json
import os
import sys

import websockets

PORT = os.environ.get("JWASH_PORT", "8381")


async def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Answer in one word: what is the capital of France?"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws") as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "sampling": {"temperature": 0.7, "max_tokens": max_tokens},
                }
            )
        )
        while True:
            frame = json.loads(await ws.recv())
            if frame["type"] == "token":
                print(frame["text"], end="", flush=True)
            elif frame["type"] == "done":
                print("\n[done]", json.dumps(frame["stats"]))
                print("[meta]", json.dumps(frame["meta"], ensure_ascii=False))
                break
            elif frame["type"] == "error":
                print("[error]", frame["message"])
                break


asyncio.run(main())
