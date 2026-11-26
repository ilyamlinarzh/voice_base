from io import BytesIO

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_full_name(person: dict | None) -> str:
    if person is None:
        return "Неизвестный спикер (требует модерации)"
    parts = [
        person.get("last_name", ""),
        person.get("first_name", ""),
        person.get("middle_name") or "",
    ]
    return " ".join(p for p in parts if p).strip() or "Без имени"


def generate_protocol_docx(
    segments: list[dict],
    persons_cache: dict[str, dict | None],
    language: str = "ru",
) -> bytes:
    doc = Document()

    title = doc.add_heading("Протокол совещания", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f"Язык: {language}    Сегментов: {len(segments)}").italic = True

    doc.add_paragraph()  

    for seg in segments:
        person_id = seg.get("person_id")
        if person_id:
            person = persons_cache.get(person_id)
            speaker_name = _format_full_name(person)
        else:
            speaker_name = "Неизвестный спикер (требует модерации)"

        timestamp = _format_timestamp(seg["start"])
        text = seg.get("text", "").strip() or "[неразборчиво]"

        p = doc.add_paragraph()
        run_header = p.add_run(f"[{timestamp}] {speaker_name}: ")
        run_header.bold = True
        if seg.get("needs_moderation"):
            run_header.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        p.add_run(text)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()