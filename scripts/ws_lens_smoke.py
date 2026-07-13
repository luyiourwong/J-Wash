import asyncio
import json
import os
import sys

import websockets

PORT = os.environ.get("JWASH_PORT", "8381")

PROMPT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Fact: The currency used in the country shaped like a boot is what? Answer in one word."
)
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 80


async def run_chat(ws, use_lens, max_tokens=MAX_TOKENS):
    await ws.send(
        json.dumps(
            {
                "type": "chat",
                "messages": [{"role": "user", "content": PROMPT}],
                "sampling": {"temperature": 0.7, "max_tokens": max_tokens},
                "lens": use_lens,
            }
        )
    )
    reading = thinking = 0
    sample_frame = None
    text = ""
    while True:
        frame = json.loads(await ws.recv())
        if frame["type"] == "frame":
            if frame["phase"] == "reading":
                reading += 1
            else:
                thinking += 1
                sample_frame = frame
        elif frame["type"] == "done":
            return frame, reading, thinking, sample_frame
        elif frame["type"] == "error":
            print("[error]", frame["message"])
            sys.exit(1)


async def main():
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        done, r, t, sample = await run_chat(ws, True)
        print(f"with lens: {done['stats']}  reading frames={r} thinking={t}")
        print(f"reply: {done['text'][:120]!r}")
        if sample:
            layer, d = sorted(sample["layers"].items(), key=lambda kv: int(kv[0]))[len(sample["layers"]) // 2]
            print(f"thinking frame pos={sample['pos']} tok={sample['tok']!r} L{layer}:",
                  [(s.strip(), round(p, 3), rk) for s, p, rk in zip(d["m_strs"][:5], d["m_p"][:5], d["m_rank"][:5])])
        done2, _, _, _ = await run_chat(ws, False)
        print(f"without lens: {done2['stats']}")


asyncio.run(main())
