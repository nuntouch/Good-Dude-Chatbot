from pathlib import Path


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class Blocklist:
    """ตรวจคำ/วลีต้องห้ามแบบ substring ไม่สนตัวพิมพ์ใหญ่เล็ก

    ผลแน่นอนและไม่เสียค่าเรียกโมเดล จึงตรวจก่อน Llama Guard เสมอ
    ไฟล์: หนึ่งบรรทัดต่อหนึ่งคำ, บรรทัดขึ้นต้นด้วย # คือคอมเมนต์
    """

    def __init__(self, path: str):
        self.terms: list[str] = []
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.terms.append(_norm(line))

    def matches(self, text: str) -> bool:
        t = _norm(text)
        return any(term in t for term in self.terms)
