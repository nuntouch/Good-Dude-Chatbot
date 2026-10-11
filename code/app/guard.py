"""ชั้นความปลอดภัย: blocklist -> Llama Guard 4 (ผ่าน OpenRouter)

ใช้ทั้งขาเข้า (ข้อความลูกค้า) และขาออก (คำตอบของบอท)
fail-closed: ถ้าเรียก Llama Guard ไม่ได้ ถือว่าตรวจไม่ผ่าน ไม่ปล่อยผ่านเงียบ ๆ
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

import httpx

from .blocklist import Blocklist
from .categories import (
    BLOCKLIST_CODE,
    BLOCKLIST_MESSAGE,
    CATEGORIES,
    OUTPUT_BLOCKED_MESSAGE,
)
from .config import Settings
from .errors import AppError

log = logging.getLogger("gooddude.guard")
_CODE_RE = re.compile(r"S\d{1,2}", re.IGNORECASE)

Direction = Literal["input", "output"]


class GuardError(Exception):
    """เรียก OpenRouter ไม่สำเร็จ หรืออ่านผลของ Llama Guard ไม่ได้"""


@dataclass
class Verdict:
    safe: bool
    message: str | None = None  # ข้อความปฏิเสธภาษาไทย (เมื่อ unsafe)
    categories: list[str] = field(default_factory=list)


def parse_verdict(raw: str) -> tuple[bool, list[str]]:
    """คืน (is_unsafe, [รหัสหมวด]) จากผล 'safe' หรือ 'unsafe\\nS1,S2'"""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        raise GuardError("ผลจาก Llama Guard ว่างเปล่า")
    first = lines[0].lower()
    if first == "safe":
        return False, []
    if first == "unsafe":
        return True, [c.upper() for c in _CODE_RE.findall(" ".join(lines[1:]))]
    raise GuardError(f"อ่านผลจาก Llama Guard ไม่ได้: {raw[:100]!r}")


def pick_blocked(codes: list[str]) -> list[str]:
    return [c for c in codes if c in CATEGORIES and CATEGORIES[c].blocked]


async def classify(
    client: httpx.AsyncClient, settings: Settings, text: str, direction: Direction
) -> tuple[bool, list[str]]:
    role = "user" if direction == "input" else "assistant"
    try:
        resp = await client.post(
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": settings.guard_model,
                "messages": [{"role": role, "content": text}],
                "temperature": 0,
            },
            timeout=settings.guard_timeout_seconds,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise GuardError(f"เรียก OpenRouter ไม่สำเร็จ: {exc}") from exc

    unsafe, codes = parse_verdict(raw or "")
    if not unsafe:
        return False, []
    blocked = pick_blocked(codes)
    return bool(blocked), blocked


async def check(
    client: httpx.AsyncClient,
    settings: Settings,
    blocklist: Blocklist,
    text: str,
    direction: Direction,
) -> Verdict:
    # 1) blocklist: ผลแน่นอน ไม่เสียค่าโมเดล (ขาออกไม่ต้องตรวจคำ injection)
    if direction == "input" and blocklist.matches(text):
        return Verdict(False, BLOCKLIST_MESSAGE, [BLOCKLIST_CODE])

    # 2) Llama Guard 4
    if not settings.guard_enabled:
        return Verdict(True)
    try:
        unsafe, codes = await classify(client, settings, text, direction)
    except GuardError as exc:
        log.error("guard failed (fail-closed): %s", exc)
        raise AppError(
            502, "guard_unavailable",
            "ระบบตรวจสอบความปลอดภัยยังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้งค่ะ",
        ) from exc

    if not unsafe:
        return Verdict(True)
    msg = CATEGORIES[codes[0]].message if direction == "input" else OUTPUT_BLOCKED_MESSAGE
    return Verdict(False, msg, codes)
