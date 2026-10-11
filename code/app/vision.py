"""อ่านรูปด้วยโมเดล vision ผ่าน OpenRouter แล้วสร้างคำตอบตามชนิดของรูป

แนวทางจาก chacharin/chatbot-with-image-bill: ให้โมเดลดึงข้อมูลเป็น JSON
ส่วนการตัดสินใจ (นับวัน, ตอบเรื่องสลิป) ทำในโค้ด ไม่ให้โมเดลคำนวณหรือรับรองเอง
"""

import base64
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from .config import Settings
from .errors import AppError, map_http_error

log = logging.getLogger("gooddude.vision")

TZ = ZoneInfo("Asia/Bangkok")
RETURN_WINDOW_DAYS = 7

# magic bytes -> mime (ตรวจจากเนื้อไฟล์ ไม่เชื่อนามสกุลหรือ content-type ที่ลูกค้าส่งมา)
_MAGIC = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
]

RETAKE_TEXT = (
    "น้องดู๊ดอ่านรูปนี้ไม่ชัดเจนค่ะ รบกวนถ่ายรูปใหม่ให้เห็นข้อความชัด ๆ "
    "(แสงพอ ไม่เบลอ เห็นครบทั้งใบ) แล้วส่งมาอีกครั้งนะคะ"
)
OFF_TOPIC_TEXT = (
    "รูปนี้ดูเหมือนไม่เกี่ยวกับสินค้าหรือการสั่งซื้อของร้านค่ะ "
    "น้องดู๊ดรับเฉพาะสลิปโอนเงิน ฉลากหรือป้ายราคาสินค้า ใบเสร็จ และรูปสินค้าที่มีตำหนิ "
    "ถ้าสอบถามเรื่องอื่น พิมพ์ถามได้เลยนะคะ"
)


def detect_mime(data: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(data: bytes, settings: Settings) -> str:
    if not data:
        raise AppError(400, "image_empty", "ไม่พบไฟล์รูปภาพ กรุณาเลือกรูปแล้วลองใหม่ค่ะ")
    if len(data) > settings.max_image_mb * 1024 * 1024:
        raise AppError(413, "image_too_large", f"ไฟล์รูปใหญ่เกิน {settings.max_image_mb:g} MB กรุณาย่อรูปแล้วลองใหม่ค่ะ")
    mime = detect_mime(data)
    if mime is None:
        raise AppError(415, "image_type", "รองรับเฉพาะไฟล์รูป JPG, PNG หรือ WEBP ค่ะ")
    return mime


def parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    for candidate in (text, braces.group(0) if braces else None):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


async def extract(
    client: httpx.AsyncClient, settings: Settings, data: bytes, mime: str
) -> dict[str, Any]:
    prompt_path = Path(settings.vision_prompt_path)
    if not prompt_path.exists():
        raise AppError(500, "vision_prompt_missing", "ไม่พบไฟล์ prompt อ่านรูป กรุณาแจ้งผู้ดูแลระบบค่ะ")
    data_uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    payload = {
        "model": settings.vision_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "อ่านรูปนี้ แล้วตอบเป็น JSON ตามรูปแบบที่กำหนด"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    }
    try:
        resp = await client.post(
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"] or ""
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        log.error("vision failed: %r", exc)
        raise map_http_error(exc, "vision") from exc

    parsed = parse_json(raw)
    if parsed is None:
        log.warning("vision returned non-JSON: %r", raw[:200])
        return {"image_type": "other", "readable": False, "fields": {}, "text_in_image": ""}
    return parsed


# ---------------------------------------------------------------------------
# ตรรกะหลังอ่านรูป (ทำในโค้ด ไม่ใช้โมเดลตัดสิน)
# ---------------------------------------------------------------------------

def normalize_date(value: Any) -> date | None:
    """แปลงวันที่ที่อ่านได้เป็น date (รองรับ พ.ศ. และรูปแบบ YYYY-MM-DD, DD/MM/YYYY)"""
    if not value:
        return None
    s = str(value).strip()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = re.search(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})", s)
        if not m:
            return None
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        if y < 100:
            y += 2500 if y > 50 else 2000
    if y > 2400:  # พ.ศ.
        y -= 543
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def days_since(d: date, today: date | None = None) -> int:
    return ((today or datetime.now(TZ).date()) - d).days


def _baht(x: Any) -> str | None:
    try:
        return f"{float(x):,.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return None


def plan(info: dict[str, Any], caption: str = "", today: date | None = None) -> tuple[str, str]:
    """คืน ("reply", ข้อความตอบสำเร็จรูป) หรือ ("rag", คำถามที่จะส่งเข้า LightRAG)"""
    f = info.get("fields") or {}
    kind = str(info.get("image_type", "other"))
    if not info.get("readable", False):
        return "reply", RETAKE_TEXT
    note = f" ข้อความเพิ่มเติมจากลูกค้า: {caption.strip()}" if caption.strip() else ""

    if kind == "slip":
        amount = _baht(f.get("amount"))
        if amount is None:
            return "reply", RETAKE_TEXT
        lines = ["น้องดู๊ดอ่านสลิปได้ดังนี้ค่ะ", f"- ยอดโอน: {amount} บาท"]
        if f.get("date"):
            lines.append(f"- วันที่: {f['date']}")
        if f.get("time"):
            lines.append(f"- เวลา: {f['time']}")
        lines.append(
            "ร้านจะตรวจสอบยอดเข้าบัญชีและยืนยันอีกครั้งนะคะ "
            "น้องดู๊ดยืนยันการชำระเงินหรือการสั่งซื้อเองไม่ได้ค่ะ"
        )
        return "reply", "\n".join(lines)

    if kind in ("product_label", "price_tag"):
        name = f.get("product_name") or "(อ่านชื่อไม่ได้)"
        size = f.get("size") or "(ไม่ระบุ)"
        price = _baht(f.get("price"))
        seen = f"ชื่อ: {name} / ขนาด: {size}" + (f" / ราคาบนป้าย: {price} บาท" if price else "")
        return "rag", (
            f"ลูกค้าส่งรูป{'ป้ายราคา' if kind == 'price_tag' else 'ฉลากสินค้า'} อ่านได้ว่า {seen}. "
            "ร้านมีสินค้านี้ในรายการหรือไม่ ราคาเท่าไหร่ "
            "ถ้าไม่มีในรายการให้บอกว่าไม่พบ และแนะนำสินค้าที่ใกล้เคียงจากรายการร้านเท่านั้น" + note
        )

    if kind == "receipt":
        d = normalize_date(f.get("date"))
        items = ", ".join(f.get("items") or []) or "(อ่านรายการไม่ได้)"
        if d is None or days_since(d, today) < 0:
            return "rag", (
                f"ลูกค้าส่งรูปใบเสร็จเพื่อขอคืนสินค้า แต่อ่านวันที่ซื้อไม่ได้ชัดเจน รายการ: {items}. "
                "ตอบตามนโยบายการคืนสินค้า บอกให้ลูกค้าถ่ายรูปใบเสร็จให้เห็นวันที่ชัดเจน "
                "และบอกว่าพนักงานเป็นผู้อนุมัติ" + note
            )
        n = days_since(d, today)
        verdict = "อยู่ภายในกำหนด" if n <= RETURN_WINDOW_DAYS else "เกินกำหนด"
        return "rag", (
            f"ลูกค้าส่งรูปใบเสร็จเพื่อขอคืนสินค้า วันที่ซื้อ {d.isoformat()} "
            f"ผ่านมาแล้ว {n} วัน ({verdict} {RETURN_WINDOW_DAYS} วัน) รายการ: {items}. "
            "ตอบตามนโยบายการคืนสินค้าของร้านตามประเภทสินค้า "
            "และบอกว่าการอนุมัติเป็นหน้าที่ของพนักงาน บอทอนุมัติเองไม่ได้" + note
        )

    if kind == "defect":
        what = f.get("note") or "สินค้าที่ลูกค้าแจ้งว่ามีตำหนิ"
        return "rag", (
            f"ลูกค้าส่งรูปสินค้ามีตำหนิ: {what}. ตอบตามนโยบายสินค้ามีตำหนิ "
            "(ต้องแจ้งภายในกี่วัน ต้องใช้อะไร) และแจ้งให้ติดต่อพนักงานเพื่อตรวจสอบ "
            "ห้ามอนุมัติคืนเงินเอง" + note
        )

    return "reply", OFF_TOPIC_TEXT
