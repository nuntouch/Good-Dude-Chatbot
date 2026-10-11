from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# code/app/config.py -> parents[2] = โฟลเดอร์โปรเจกต์ (มี prompts/ และ knowledge/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CODE_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenRouter (ใช้ทั้ง Llama Guard และโมเดลอ่านรูป)
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    guard_model: str = "meta-llama/llama-guard-4-12b"
    vision_model: str = "openai/gpt-4o-mini"

    # LightRAG
    lightrag_url: str = "http://localhost:9621"
    lightrag_api_key: str = ""
    rag_mode: str = "mix"
    rag_top_k: int = 20

    # พฤติกรรมของบอท
    guard_enabled: bool = True
    request_timeout_seconds: float = 60.0
    guard_timeout_seconds: float = 20.0
    max_message_chars: int = 1000
    max_history_turns: int = 6
    max_image_mb: float = 5.0

    # path ของไฟล์ที่แก้ได้โดยไม่ต้องแก้โค้ด
    rules_path: str = str(_PROJECT_ROOT / "prompts" / "bot_rules.md")
    vision_prompt_path: str = str(_PROJECT_ROOT / "prompts" / "vision_system.md")
    blocklist_path: str = str(_CODE_ROOT / "app" / "blocklist.txt")
    static_dir: str = str(_CODE_ROOT / "static")

    cors_allowed_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
