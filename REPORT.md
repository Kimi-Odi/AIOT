# AI Interview Officer – Technical Report

## System Overview
- **UI/Orchestration**: `app.py` (Streamlit) drives session state, audio recording/upload, question flow, grading calls, plotting, and report downloads.
- **Speech Pipeline**: Whisper STT (`speech_to_text`), speech analytics (WPM, silence ratio, volume stability, filler ratio via `librosa`/`soundfile`), and GPT-4o-mini TTS for playback.
- **LLM Scoring**: `grader.py` scores each Q&A (6 dimensions) using `gpt-4.1-mini`, adjusts communication/structure with speech features, and generates an overall summary.
- **Resume Parsing**: `resume_parser.py` extracts PDF text (PyPDF2) and asks OpenAI to return structured JSON (skills/projects/experience/education) plus raw text.
- **RAG Context**: `rag_retriever.py` (loaded via `SimpleRAG` in `app.py`) keyword-matches queries against `knowledge/*.md` to surface role-relevant snippets.
- **Persistence**: `db.py` initializes and writes to SQLite (`interview.db`): candidates, interviews, per-question QA, and per-interview scores.
- **Export**: `pdf_export.py` builds styled PDFs (ReportLab/matplotlib figures), `html_export.py` renders HTML, and the app also offers Markdown download.
- **Assets**: `static/style.css` customizes the Streamlit UI; `knowledge/` holds seed domain notes.

## Data Flow (Happy Path)
1. User sets candidate/job role and (optionally) uploads a PDF resume → parsed to structured JSON and stored in session.
2. User records or types an answer → Whisper transcription → speech feature analysis.
3. Q&A + speech features go to `grade_single_qa` → per-question scores/feedback → aggregated in `grade_interview`.
4. Results and charts render in Streamlit; records persist via `save_*` helpers.
5. Report Markdown is assembled; PDF/HTML/MD downloads are offered (including radar plots and optional history comparison).

## Dependencies
- Core: `streamlit`, `openai`, `python-dotenv`, `numpy`, `pandas`, `matplotlib`
- Audio: `faster-whisper`, `pydub`, `librosa`, `soundfile`
- Parsing/Export: `PyPDF2`, `reportlab`, `jinja2`
- Storage: `sqlite3` (standard library)

## Operational Notes
- Environment: requires `OPENAI_API_KEY` in `.env`.
- Models are hard-coded; update in `app.py`/`grader.py` if you change tiers or endpoints.
- Database is local and file-based; handle `interview.db` with care if storing real candidate data.
- Knowledge base is keyword-matching; enrich `knowledge/` to steer relevance.

## Testing & Validation (manual)
- Launch with `streamlit run app.py`, verify: resume upload → parsed JSON, mic recording → Whisper transcription, scoring → metrics + radar chart, history selection → comparison chart, downloads → valid MD/PDF/HTML files.
- Audio checks: confirm ffmpeg availability; ensure WPM/silence values change with speaking style.
