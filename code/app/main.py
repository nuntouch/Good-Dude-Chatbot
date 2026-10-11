import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import guard, rag, vision
from .blocklist import Blocklist
from .categories import BLOCKLIST_CODE, BLOCKLIST_MESSAGE
from .config import get_settings
from .errors import AppError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gooddude")

settings = get_settings()
blocklist = Blocklist(settings.blocklist_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    yield
    await app.state.http.aclose()


app = FastAPI(title="Good Dude Pet Shop Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})


# ---------------------------------------------------------------------------
# โครงสร้างข้อมูล
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[Turn] = []


class ChatResponse(BaseModel):
    reply: str
    blocked: bool = False  # ถูกชั้นความปลอดภัยปฏิเสธ
    stage: Literal["input", "output", "image", "none"] = "none"  # ปฏิเสธที่ขั้นไหน
    categories: list[str] = []
    fallback: bool = False  # ตอบด้วยข้อความ fallback (นอกคลังความรู้)
    image_type: str | None = None
    elapsed_ms: int = 0


class CheckRequest(BaseModel):
    message: str = Field(min_length=1)
    direction: Literal["input", "output"]


def _clean_history(history: list[Turn]) -> list[dict[str, str]]:
    turns = history[-(settings.max_history_turns * 2):]
    return [{"role": t.role, "content": t.content} for t in turns if t.content.strip()]


def _validate_message(text: str) -> str:
    text = text.strip()
    if not text:
        raise AppError(400, "empty_message", "กรุณาพิมพ์ข้อความก่อนส่งค่ะ")
    if len(text) > settings.max_message_chars:
        raise AppError(
            413, "message_too_long",
            f"ข้อความยาวเกิน {settings.max_message_chars} ตัวอักษร กรุณาย่อข้อความแล้วลองใหม่ค่ะ",
        )
    return text


async def _answer(question: str, history: list[dict[str, str]]) -> tuple[str, bool, list[str]]:
    """RAG + ตรวจขาออก คืน (ข้อความ, ถูกบล็อกขาออกไหม, หมวด)"""
    client = app.state.http
    answer = await rag.query(client, settings, question, history)
    if answer == rag.FALLBACK_TEXT:
        return answer, False, []  # ข้อความตายตัวของร้าน ไม่ต้องตรวจซ้ำ
    verdict = await guard.check(client, settings, blocklist, answer, "output")
    if not verdict.safe:
        return verdict.message or "", True, verdict.categories
    return answer, False, []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """ตรวจว่าแอปและ LightRAG พร้อมหรือไม่ (UI ใช้แสดงสถานะ)"""
    rag_ok = False
    try:
        r = await app.state.http.get(f"{settings.lightrag_url.rstrip('/')}/health", timeout=3)
        rag_ok = r.status_code == 200
    except httpx.HTTPError:
        pass
    return {"status": "ok", "rag": "ok" if rag_ok else "down", "guard": "on" if settings.guard_enabled else "off"}


@app.post("/api/check")
async def check_endpoint(req: CheckRequest):
    """เรียกชั้นความปลอดภัยโดยตรง (ใช้ทดสอบด้วย Postman)"""
    v = await guard.check(app.state.http, settings, blocklist, req.message, req.direction)
    return {
        "result": "safe" if v.safe else "unsafe",
        "direction": req.direction,
        "categories": v.categories,
        "msg": req.message if v.safe else v.message,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    text = _validate_message(req.message)

    verdict = await guard.check(app.state.http, settings, blocklist, text, "input")
    if not verdict.safe:
        return ChatResponse(
            reply=verdict.message or BLOCKLIST_MESSAGE, blocked=True, stage="input",
            categories=verdict.categories, elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )

    reply, out_blocked, cats = await _answer(text, _clean_history(req.history))
    return ChatResponse(
        reply=reply, blocked=out_blocked, stage="output" if out_blocked else "none",
        categories=cats, fallback=reply == rag.FALLBACK_TEXT,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.post("/api/chat/image", response_model=ChatResponse)
async def chat_image(
    file: UploadFile = File(...),
    message: str = Form(""),
) -> ChatResponse:
    t0 = time.perf_counter()
    client = app.state.http

    caption = message.strip()
    if len(caption) > settings.max_message_chars:
        raise AppError(413, "message_too_long", "ข้อความยาวเกินกำหนด กรุณาย่อข้อความแล้วลองใหม่ค่ะ")

    data = await file.read(int(settings.max_image_mb * 1024 * 1024) + 1)
    mime = vision.validate_image(data, settings)

    if caption:
        v = await guard.check(client, settings, blocklist, caption, "input")
        if not v.safe:
            return ChatResponse(
                reply=v.message or BLOCKLIST_MESSAGE, blocked=True, stage="input",
                categories=v.categories, elapsed_ms=int((time.perf_counter() - t0) * 1000),
            )

    info = await vision.extract(client, settings, data, mime)
    f = info.get("fields") or {}

    # ข้อความที่ฝังอยู่ในรูปอาจเป็นคำสั่งโจมตี ตรวจ blocklist ก่อนใช้ต่อ
    embedded = " ".join(str(x) for x in (info.get("text_in_image"), f.get("product_name"), f.get("note")) if x)
    if embedded and blocklist.matches(embedded):
        return ChatResponse(
            reply=BLOCKLIST_MESSAGE, blocked=True, stage="image", categories=[BLOCKLIST_CODE],
            image_type=str(info.get("image_type")), elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )

    kind, content = vision.plan(info, caption)
    image_type = str(info.get("image_type", "other"))
    if kind == "reply":
        return ChatResponse(
            reply=content, image_type=image_type, elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )

    reply, out_blocked, cats = await _answer(content, [])
    return ChatResponse(
        reply=reply, blocked=out_blocked, stage="output" if out_blocked else "none",
        categories=cats, fallback=reply == rag.FALLBACK_TEXT, image_type=image_type,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# หน้าเว็บ
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(f"{settings.static_dir}/index.html")
