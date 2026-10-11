FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY static ./static
# prompts/ ถูก mount จากโฟลเดอร์โปรเจกต์ผ่าน docker-compose (แก้ไฟล์แล้วมีผลทันที)
ENV RULES_PATH=/app/prompts/bot_rules.md \
    VISION_PROMPT_PATH=/app/prompts/vision_system.md
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
