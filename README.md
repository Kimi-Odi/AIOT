# AI Interview Officer (AIOT)

A Streamlit web app for running mock technical interviews. It uses OpenAI for speech-to-text (Whisper), text-to-speech, and scoring/summarization, plus a lightweight RAG knowledge base and SQLite persistence to grade answers, analyze speaking performance, and export reports.

## Features
- Voice loop: record answers → Whisper transcription → speech analytics (WPM, silence ratio, volume stability, filler words) → optional TTS playback.
- Resume intelligence: PDF resume parsing via OpenAI to extract skills/experience and attach raw text for context.
- RAG assistance: keyword-matching retriever over `knowledge/*.md` to bias questions and feedback toward the target role.
- Auto-grading: per-question scoring (technical, communication, structure, relevance, problem solving, growth potential) and overall summary.
- History: SQLite (`interview.db`) stores candidates, interviews, QA pairs, and scores for comparison across sessions.
- Exports: download Markdown/PDF/HTML reports with charts; audio metrics surfaced in the UI.

## Prerequisites
- Python 3.9+ and pip
- ffmpeg installed and on PATH (required by `pydub`/`librosa` for audio)
- OpenAI API key in `.env`:
  ```
  OPENAI_API_KEY=sk-...
  ```

## Installation
```bash
# from repository root
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
The app uses `interview.db` in the working directory; first run initializes tables automatically.

## Typical Flow
1. Enter candidate ID/name and target job role.
2. (Optional) Upload a PDF resume to extract skills/projects/experience.
3. Start the interview, record answers (or type them), and submit.
4. Review per-question scores, AI feedback, speech metrics, and radar chart; compare against past sessions if selected.
5. Download the report in Markdown/PDF/HTML.

## Configuration Notes
- Models: `whisper-1`, `gpt-4o-mini-tts`, `gpt-4.1-mini` (update in code if you prefer other models).
- Knowledge base: add/edit `.md` or `.txt` files in `knowledge/` to tune retrieval.
- Styling: custom CSS in `static/style.css`.
- Database: schema lives in `db.py` (candidates, interviews, qa_records, scores). Remove or rotate `interview.db` if you want a clean slate.

## Troubleshooting
- Audio errors: ensure ffmpeg is installed and microphone permissions are granted.
- Missing API key: set `OPENAI_API_KEY` in `.env`.
- PDF parsing: only PDF resumes are supported; non-text PDFs may yield weak extraction quality.
