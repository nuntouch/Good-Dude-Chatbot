# Good Dude Pet Shop Chatbot (น้องดู๊ด)

แชทบอทร้านขายอาหารและของใช้สุนัข-แมว (ข้อมูลร้านจำลอง) สำหรับโปรเจกต์วิชา 06048308 Intelligent Chatbot Development

```
เบราว์เซอร์ (static/index.html)
   │  POST /api/chat | /api/chat/image
   ▼
FastAPI (app/main.py)
   ├─ ตรวจขาเข้า: blocklist.txt → Llama Guard 4 (OpenRouter)   [fail-closed]
   ├─ ข้อความ → LightRAG /query (mode=mix, user_prompt = prompts/bot_rules.md)
   ├─ รูป → vision model อ่านเป็น JSON → โค้ดตัดสิน (สลิป/นับวันคืนสินค้า) → LightRAG
   └─ ตรวจขาออก: Llama Guard 4 → ตอบลูกค้า
```

## โครงสร้าง
| path | หน้าที่ |
|---|---|
| `app/` | backend (config, guard, rag, vision, errors, main) |
| `app/blocklist.txt` | คำต้องห้าม/คำสั่งแทรก แก้ได้ทันที ต้อง restart |
| `static/index.html` | หน้าแชท (สถานะกำลังโหลด, error + ปุ่มลองใหม่, แนบรูป) |
| `scripts/ingest.py` | นำ `../knowledge/*.md` เข้า LightRAG |
| `scripts/run_tests.py` | รันชุดทดสอบ บันทึกตารางผล + เวลาตอบ |
| `tests/cases.json` | 10 คำถาม + กรณีความปลอดภัย พร้อมเกณฑ์จับคำสำคัญ |
| `tests/test_unit.py` | unit test แบบ mock (ไม่ใช้คีย์จริง) |
| `../prompts/bot_rules.md`, `vision_system.md` | กฎบอท / prompt อ่านรูป (อ่านใหม่ทุกคำขอ แก้แล้วเห็นผลทันที) |

## วิธีรัน
1. `cp .env.example .env` แล้วใส่ `OPENROUTER_API_KEY` และตั้ง `LIGHTRAG_API_KEY`
2. `docker compose up -d --build` (เปิด LightRAG พอร์ต 9621 และแอปพอร์ต 8000)
3. `pip install -r requirements.txt` แล้ว `python scripts/ingest.py` รอจน busy=False
4. เปิด http://localhost:8000
5. ทดสอบ: `python scripts/run_tests.py --suite text` (และ `safety`, `image`) ผลอยู่ใน `tests/results/`

รันแอปนอก Docker: `uvicorn app.main:app --port 8000` (ต้องมี LightRAG รันอยู่แล้ว และ `LIGHTRAG_URL=http://localhost:9621`)

## API (ทดสอบด้วย Postman ได้)
- `GET /api/health` สถานะแอป + LightRAG
- `POST /api/chat` `{"message": "...", "history": [{"role":"user","content":"..."}]}`
- `POST /api/chat/image` multipart: `file` (JPG/PNG/WEBP ≤ 5 MB), `message` (ไม่บังคับ)
- `POST /api/check` `{"message": "...", "direction": "input|output"}` เรียกชั้นความปลอดภัยโดยตรง

ข้อผิดพลาดทุกกรณีตอบ `{"error": {"code", "message"}}` ข้อความภาษาไทย (400/413/415 ข้อมูลผิด, 429 ถูกจำกัด, 502 guard ใช้ไม่ได้, 503/504 LightRAG หรือโมเดลไม่พร้อม/ช้า)

## หมายเหตุ
- ไม่ใช้ Vercel และไม่เชื่อม LINE ตามที่ตกลง ประวัติสนทนาเก็บฝั่งเบราว์เซอร์ เซิร์ฟเวอร์ไม่เก็บ state
- ข้อความที่ฝังในรูปจะถูกตรวจ blocklist ก่อนใช้ต่อ กัน prompt injection ผ่านรูป
- สลิป: บอทอ่านยอด/วันที่/เวลาแล้วบอกว่าร้านต้องตรวจยอดเอง ไม่ยืนยันการชำระเงินเด็ดขาด
- ยังไม่เคยรันกับ OpenRouter/LightRAG จริง: ถ้า embedding ผ่าน OpenRouter ไม่ได้ ให้ลองเปลี่ยน `EMBEDDING_USE_BASE64` หรือชื่อโมเดลใน `docker-compose.yml`
