# AI Interview Officer – Specification Sheet

## Product Scope
- Purpose: Deliver a browser-based mock interview platform with automated scoring, speech analysis, resume awareness, lightweight retrieval-augmented context, and exportable reports.
- Target users: Interview coaches, candidates practicing technical interviews, and recruiters needing quick screening simulations.
- Platforms: Desktop browser (Streamlit UI). Audio capture requires microphone access.

## Functional Requirements
- Session setup:
  - Input candidate ID/name and target job role.
  - Optional: upload a PDF resume for parsing.
- Question/answer flow:
  - Record audio answers via mic or type answers.
  - Transcribe audio to text using Whisper; display transcript for confirmation.
  - Analyze speech features: words per minute, silence ratio, volume stability, filler ratio.
- Scoring:
  - For each Q&A, grade technical, communication, structure, relevance, problem solving, growth potential (1–5 scale; decimals allowed).
  - Adjust communication/structure scores using speech features.
  - Generate per-question feedback and an overall interview summary.
- Retrieval (RAG):
  - Keyword-match role-specific snippets from `knowledge/*.md` based on job role and query to inform feedback.
- Persistence:
  - Store candidates, interviews, per-question QA, and scores in SQLite (`interview.db`).
  - Allow selecting a past interview to compare scores in charts.
- Reporting & Exports:
  - Generate Markdown report with summary, per-question scores/feedback, speech metrics, and improvement tips.
  - Export PDF and HTML versions; include radar charts and optional comparison plots.
- UI:
  - Streamlit layout with tabs/columns, metrics, charts (matplotlib radar), download buttons, and custom CSS (`static/style.css`).

## Non-Functional Requirements
- Performance: RAG uses local keyword matching; latency primarily from OpenAI API calls (STT, TTS, grading). Keep per-call payloads concise.
- Availability: Single-instance Streamlit app; state is per-session in memory plus SQLite.
- Security/Privacy: API key loaded from `.env`; no in-app authentication. Do not store sensitive resumes in shared environments without consent.
- Internationalization: UI text mixes English/Traditional Chinese; fonts configured to render CJK in matplotlib/PDF. Whisper supports multilingual transcription.
- Extensibility: Knowledge base is file-backed; models are configurable constants in code; DB schema centralized in `db.py`.

## Architecture
- Frontend/Orchestration: `app.py` (Streamlit) manages session state, audio recording/upload, question flow, grading, plotting, and report assembly.
- AI/LLM:
  - STT: `whisper-1`
  - TTS: `gpt-4o-mini-tts`
  - Grading/Summary: `gpt-4.1-mini`
  - Embeddings (for RAG retriever variant): `text-embedding-3-small`
- Resume Parsing: `resume_parser.py` (PyPDF2 + OpenAI JSON extraction).
- RAG: `rag_retriever.py` builds embeddings over `knowledge/*.md` paragraphs and retrieves top matches filtered by job type.
- Data & Persistence: `db.py` initializes and accesses SQLite tables (candidates, interviews, qa_records, scores).
- Exports: `pdf_export.py` (+ `pdf_styles.py`) for PDF with CJK-friendly fonts and embedded plots; `html_export.py` for HTML; Markdown assembled in `app.py`.
- Assets: `static/style.css` for theming; `knowledge/` for domain notes.

## Data Model (SQLite)
- `candidates(id PK, candidate_id UNIQUE, name)`
- `interviews(id PK, candidate_id, job_role, timestamp, summary)`
- `qa_records(id PK, interview_id FK, question, answer)`
- `scores(id PK, interview_id FK, technical, communication, structure, relevance, problem_solving, growth_potential)`
- Database file: `interview.db` in working directory; initialized on app start via `init_db()`.

## External Inputs/Outputs
- Inputs: Microphone audio (uploaded/recorded), PDF resume upload, manual text answers, job role selection.
- Outputs: On-screen metrics/charts, downloadable Markdown/PDF/HTML reports, stored DB records, optional TTS audio playback.

## User Flow (Happy Path)
1. User opens app, enters candidate ID/name and job role.
2. (Optional) Uploads PDF resume; parsed into structured JSON and stored in session.
3. Starts interview; for each question, records audio or types answer.
4. Whisper transcribes; speech features computed; LLM grades Q&A with feedback.
5. After completion, app aggregates scores, generates radar chart, and compares to a selected historical interview if chosen.
6. User downloads report (MD/PDF/HTML) and optionally plays back TTS.

## Constraints & Assumptions
- Requires `OPENAI_API_KEY` in `.env`.
- ffmpeg must be installed and on PATH for audio handling.
- Resume parsing supports PDF only; image-based PDFs may yield poor text.
- Knowledge base retrieval is keyword/embedding-based, not a full semantic RAG service; quality depends on `knowledge/` content.
- Single-user session memory; not multi-tenant.

## Installation & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Validation Checklist (manual)
- API key loaded; app starts without errors.
- Resume upload → structured JSON fields present.
- Mic recording → Whisper transcript displayed; metrics (WPM, silence, fillers, volume stability) populated.
- Per-question scores and feedback rendered; overall summary present.
- Radar chart displays; comparison chart appears when a historical interview is selected.
- Downloads: Markdown, PDF, and HTML files open successfully with correct content and charts.

## Future Improvements
- Add authentication and role-based access.
- Swap keyword RAG with vector search and caching.
- Support non-PDF resumes (DOCX/HTML) and OCR for scanned PDFs.
- Add configurable question banks and adaptive difficulty.
- Add automated tests (unit for parsers/DB, integration for grading pipeline via mocks).
