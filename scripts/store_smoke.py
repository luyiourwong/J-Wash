import asyncio
import json
import os
import urllib.request

import websockets

PORT = os.environ.get("JWASH_PORT", "8381")
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path) as res:
        return json.load(res)


def get_text(path):
    with urllib.request.urlopen(BASE + path) as res:
        return res.read().decode("utf-8")


async def chat(ws, payload):
    await ws.send(json.dumps(dict(payload, type="chat")))
    persisted = None
    frames = 0
    while True:
        frame = json.loads(await ws.recv())
        if frame["type"] == "persisted":
            persisted = frame
        elif frame["type"] == "frame":
            frames += 1
        elif frame["type"] == "done":
            return persisted, frame, frames
        elif frame["type"] == "error":
            raise SystemExit("error: " + frame["message"])


async def main():
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        p1, d1, f1 = await chat(ws, {
            "content": "What is the capital of Italy? One word only.",
            "system": "Answer very concisely.",
            "sampling": {"max_tokens": 30},
            "lens": True,
        })
        cid = d1["conversation_id"]
        print(f"conv {cid} · user #{p1['user_message_id']} · assistant #{d1['message_id']} · {f1} frames · {d1['text']!r}")

        p2, d2, f2 = await chat(ws, {
            "conversation_id": cid,
            "parent_id": d1["message_id"],
            "content": "And Spain's?",
            "sampling": {"max_tokens": 30},
            "lens": True,
        })
        print(f"follow-up: user #{p2['user_message_id']} · assistant #{d2['message_id']} · {f2} frames · {d2['text']!r}")

        p3, d3, f3 = await chat(ws, {
            "conversation_id": cid,
            "parent_id": p1["user_message_id"],
            "content": None,
            "sampling": {"max_tokens": 30},
            "lens": False,
        })
        print(f"regeneration (branch): assistant #{d3['message_id']} · {d3['text']!r}")

    tree = get(f"/api/conversations/{cid}")
    print("tree:", [(m["id"], m["parent_id"], m["role"], m["has_frames"]) for m in tree["messages"]])

    replay = get(f"/api/messages/{d1['message_id']}/frames")
    sample_layer = str(replay["layers"][len(replay["layers"]) // 2])
    print(f"replay: {len(replay['frames'])} frames · layers {replay['layers'][0]}-{replay['layers'][-1]} · "
          f"last m_strs L{sample_layer}: {replay['frames'][-1]['layers'][sample_layer]['m_strs'][:4]}")

    search = get("/api/conversations?query=Italy")
    print("FTS search:", [(c["id"], c["snippet"]) for c in search["conversations"]])

    md = get_text(f"/api/conversations/{cid}/export?format=markdown&frames=1")
    print("export markdown:", len(md), "chars, excerpt:", md.splitlines()[0])


asyncio.run(main())
