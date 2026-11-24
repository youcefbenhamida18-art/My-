"""Find lessons under lessons/ and prepare their files for the LLM.

Layout (see README_AR.md):
  lessons/<course>/td/                shared past TD / exams for the whole course (optional)
  lessons/<course>/<lesson>/          ONE lesson = lecture files (+ optional td/ folder, notes.txt)
  lessons/<course>/<lecture>.pdf      a single file = ONE lesson
  lessons/<lecture>.pdf               a single file = ONE lesson (course "General")
"""
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from common import ROOT, WORK, log, sha

SUPPORTED = {".pdf", ".pptx", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
TD_DIRS = {"td", "tds", "tp", "exercises", "exercices", "exos", "exams", "examens", "annales", "corriges", "corrigés"}
LECTURE_DIRS = {"cours", "lecture", "lectures", "course"}
TD_FILE_RE = re.compile(
    r"(^|[^a-z])(td|tds|tp|exo|exos|exercice|exercices|exercise|exercises|exam|examen|examens|"
    r"controle|contrôle|annale|annales|corrige|corrigé|corriges)([^a-z]|$)",
    re.I,
)
MAX_FILES = 14
MAX_TOTAL_MB = 80


@dataclass
class LessonFile:
    path: Path
    role: str  # "lecture" | "td"
    kind: str = ""  # pdf | text | image
    pdf: Path | None = None  # PDF to send (original or converted)
    text: str = ""  # extracted text (for txt/pptx/docx fallback)
    image: Path | None = None
    pages: int = 0
    fp: str = ""  # content fingerprint

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Lesson:
    key: str  # posix path relative to repo root
    course: str
    name: str
    files: list = field(default_factory=list)  # list[LessonFile]
    notes: str = ""

    @property
    def title_hint(self) -> str:
        return f"{self.course} - {self.name}".strip(" -")


def _rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def _is_td_name(path: Path) -> bool:
    return bool(TD_FILE_RE.search(path.stem))


def _role(path: Path, base: Path) -> str:
    parts = [x.lower() for x in path.relative_to(base).parts[:-1]]
    if any(x in TD_DIRS for x in parts):
        return "td"
    if any(x in LECTURE_DIRS for x in parts):
        return "lecture"
    return "td" if _is_td_name(path) else "lecture"


def _supported(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith(".") and p.name.lower() != "readme.md"


def _read_notes(folder: Path) -> str:
    for n in ("notes.txt", "notes.md", "instructions.txt"):
        f = folder / n
        if f.exists():
            return f.read_text(encoding="utf-8", errors="ignore").strip()[:3000]
    return ""


def _td_bank(course_dir: Path) -> list:
    out = []
    for d in course_dir.iterdir():
        if d.is_dir() and d.name.lower() in TD_DIRS:
            out += [(p, "td") for p in sorted(d.rglob("*")) if _supported(p)]
    out += [(p, "td") for p in sorted(course_dir.iterdir()) if _supported(p) and _is_td_name(p)]
    return out


def discover(root: Path) -> list:
    lessons = []
    if not root.exists():
        return lessons
    for p in sorted(root.iterdir()):
        if p.name.startswith("."):
            continue
        if _supported(p):
            lessons.append(Lesson(_rel(p), "General", p.stem, [LessonFile(p, "lecture")]))
        elif p.is_dir():
            bank = _td_bank(p)
            for q in sorted(p.iterdir()):
                if q.name.startswith(".") or q.name.lower() in TD_DIRS:
                    continue
                if _supported(q):
                    if _is_td_name(q):
                        continue  # part of the course TD bank
                    files = [LessonFile(q, "lecture")] + [LessonFile(a, r) for a, r in bank]
                    lessons.append(Lesson(_rel(q), p.name, q.stem, files, ""))
                elif q.is_dir():
                    files = []
                    for f in sorted(q.rglob("*")):
                        if _supported(f) and f.name.lower() not in ("notes.txt", "notes.md", "instructions.txt"):
                            files.append(LessonFile(f, _role(f, q)))
                    if not any(x.role == "lecture" for x in files):
                        continue
                    files += [LessonFile(a, r) for a, r in bank]
                    lessons.append(Lesson(_rel(q), p.name, q.name, files, _read_notes(q)))
    return lessons


# ----------------------------------------------------------------------------- preparation
def _count_pages(pdf: Path) -> int:
    try:
        import pypdf

        return len(pypdf.PdfReader(str(pdf)).pages)
    except Exception:
        return 0


def _office_to_pdf(src: Path, outdir: Path) -> Path | None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(src)],
                       check=True, capture_output=True, timeout=300)
        pdf = outdir / (src.stem + ".pdf")
        return pdf if pdf.exists() else None
    except Exception as e:
        log(f"  LibreOffice conversion failed for {src.name}: {e}")
        return None


def _pptx_text(p: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(p))
    out = []
    for i, slide in enumerate(prs.slides, 1):
        out.append(f"--- Slide {i} ---")
        for sh in slide.shapes:
            if getattr(sh, "has_text_frame", False) and sh.text_frame.text.strip():
                out.append(sh.text_frame.text.strip())
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    out.append(" | ".join(c.text.strip() for c in row.cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            out.append("[notes] " + slide.notes_slide.notes_text_frame.text.strip())
    return "\n".join(out)


def _docx_text(p: Path) -> str:
    import docx

    d = docx.Document(str(p))
    out = [para.text for para in d.paragraphs if para.text.strip()]
    for t in d.tables:
        for row in t.rows:
            out.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(out)


def prepare(lesson: Lesson) -> Lesson:
    """Fill kind/pdf/text/image/pages/fp for every file (and drop what cannot be used)."""
    work = WORK / "prep" / sha(lesson.key)
    ready, total = [], 0.0
    # lecture files first, then TD
    ordered = sorted(lesson.files, key=lambda f: (f.role != "lecture", f.path.name.lower()))
    for f in ordered:
        if len(ready) >= MAX_FILES:
            log(f"  skipping {f.name}: more than {MAX_FILES} files")
            continue
        size = f.path.stat().st_size / 1e6
        if total + size > MAX_TOTAL_MB:
            log(f"  skipping {f.name}: total size limit")
            continue
        ext = f.path.suffix.lower()
        f.fp = sha(f.path.read_bytes())
        try:
            if ext == ".pdf":
                f.kind, f.pdf = "pdf", f.path
            elif ext in (".pptx", ".docx"):
                pdf = _office_to_pdf(f.path, work)
                if pdf:
                    f.kind, f.pdf = "pdf", pdf
                else:
                    f.kind = "text"
                    f.text = _pptx_text(f.path) if ext == ".pptx" else _docx_text(f.path)
                    log(f"  {f.name}: no LibreOffice, using text only (pictures are lost; export to PDF for best results)")
            elif ext in (".txt", ".md"):
                f.kind, f.text = "text", f.path.read_text(encoding="utf-8", errors="ignore")
            elif ext in IMAGE_MIME:
                f.kind, f.image = "image", f.path
            else:
                continue
            if f.kind == "pdf":
                f.pages = _count_pages(f.pdf)
        except Exception as e:
            log(f"  could not read {f.name}: {e}")
            continue
        total += size
        ready.append(f)
    lesson.files = ready
    return lesson


def catalog(lesson: Lesson) -> str:
    lines = []
    for i, f in enumerate(lesson.files, 1):
        role = "LECTURE" if f.role == "lecture" else "PAST TD / EXAM (exercises)"
        extra = f", {f.pages} pages" if f.pages else ""
        lines.append(f"FILE {i}: {f.name} - {role} - {f.kind}{extra}")
    return "\n".join(lines)


def fingerprint(lesson: Lesson) -> str:
    return sha(*[f.fp + f.role for f in lesson.files], lesson.notes)
