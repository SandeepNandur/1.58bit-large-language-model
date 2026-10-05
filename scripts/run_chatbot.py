"""Run the optional FastAPI mock chatbot."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from sugumi.inference import SugumiEngine


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


app = FastAPI(title="Sugumi Mock Chat API")
engine = SugumiEngine(mock=True)


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> dict[str, object]:
    reply = engine.chat([message.model_dump() for message in request.messages])
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sugumi mock chat API.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run the mock API (required; real inference is not implemented)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not args.mock:
        parser.error("only mock mode is available; pass --mock")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
