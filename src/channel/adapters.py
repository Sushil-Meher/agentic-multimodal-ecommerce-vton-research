"""Where replies go. The WhatsApp Cloud API adapter (Slice 2) plugs in here with the same method."""
import os
from typing import Protocol

from .handler import OutgoingMessage


class ChannelAdapter(Protocol):
    def send(self, to: str, message: OutgoingMessage) -> None: ...


class ConsoleAdapter:
    """Local test channel: prints what a customer would see (and opens photos in the viewer)."""

    def __init__(self, open_photos: bool = False):
        self.open_photos = open_photos

    def send(self, to: str, message: OutgoingMessage) -> None:
        if message.kind == "image":
            print(f"   [photo: {message.image_path}]\n   {message.text}")
            if self.open_photos and hasattr(os, "startfile"):
                os.startfile(message.image_path)  # Windows: default image viewer
        else:
            print("\n".join("   " + line for line in message.text.split("\n")))
