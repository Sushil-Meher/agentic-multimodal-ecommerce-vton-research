"""Chat with the sales agent from the terminal, as a WhatsApp customer would.

  python -m src.channel.local_chat [--phone +919999900001]
  /photo <file>   send a photo      /new   start a fresh conversation      quit   leave
Needs Docker (database) and Ollama. Photo search also needs the embedding service.
"""
import argparse
import os
import sys
from pathlib import Path

from ..agent.graph import build_graph
from ..api.db import engine
from .adapters import ConsoleAdapter
from .handler import IncomingMessage, handle_incoming

TEST_PHONE = "+919999900001"  # a made-up number; real customers arrive in Slice 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", default=TEST_PHONE)
    parser.add_argument("--no-open-photos", action="store_true", help="only print photo paths")
    args = parser.parse_args()

    # development only: the catalog is still DRAFT, so unpublished sarees must be searchable
    os.environ.setdefault("AGENT_INCLUDE_DRAFT", "1")
    graph, adapter = build_graph(), ConsoleAdapter(open_photos=not args.no_open_photos)
    print("Local WhatsApp test chat. /photo <file>, /new, quit.")
    fresh = False
    while True:
        try:
            line = input("you: ").strip()
        except EOFError:
            break
        if not line or line.lower() in {"quit", "exit"}:
            break
        if line == "/new":
            fresh = True
            print("   (new conversation)")
            continue
        if line.startswith("/photo"):
            parts = line.split(maxsplit=2)
            path = Path(parts[1].strip('"')) if len(parts) > 1 else None
            if not path or not path.is_file():
                print("   usage: /photo <file> [caption]")
                continue
            msg = IncomingMessage(sender=args.phone, kind="image", image_bytes=path.read_bytes(), text=parts[2] if len(parts) > 2 else "")
        else:
            msg = IncomingMessage(sender=args.phone, text=line)
        for reply in handle_incoming(engine, graph, msg, new_conversation=fresh):
            adapter.send(args.phone, reply)
        fresh = False


if __name__ == "__main__":
    sys.exit(main())
