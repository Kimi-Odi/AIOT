# ============================================================
# PART 1 — Imports、初始化、資料庫、語音（Whisper/TTS）、RAG
# ============================================================

import os
import json
import io
import hashlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import librosa
import soundfile as sf

# 自訂模組
from resume_parser import parse_resume
from grader import grade_interview
from pdf_export import export_pdf
from html_export import export_html
from db import (
    init_db,
    save_candidate,
    save_interview,
    save_qa,
    save_scores,
    get_interviews,
    get_scores,
    get_qa,
)

# ====== 初始化 ======
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("請在 .env 中設定 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ====== 字型設定 ======
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ====== 初始化資料庫 ======
init_db()

# ============================================================
# ------------- 語音功能（Whisper + TTS） ---------------------
# ============================================================


def speech_to_text(file):
    """
    Whisper 語音辨識（回傳 Python dict，需要 verbose_json）
    """
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        response_format="verbose_json"
    )
    return resp.model_dump()   # ⭐ 回傳 dict（不是 Transcription 物件）


def synthesize_speech(text: str) -> bytes:
    """
    TTS — 文字轉語音
    """
    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
        )
        return resp.read()
    except Exception as e:
        st.error(f"TTS 錯誤：{e}")
        return None


# ============================================================
# ----------- 語音特徵分析：WPM / Silence / Volume / Fillers ----
# ============================================================

FILLERS = ["嗯", "呃", "那個", "就是", "like", "you know"]


def analyze_speech_features(whisper_resp, audio_bytes):
    """
    回傳 dict：
    {
      wpm,
      silence_ratio,
      volume_stability,
      filler_ratio
    }
    """

    result = {}

    # -------------------------
    # 1) 語速（WPM）
    # -------------------------
    total_words = len(whisper_resp["text"].split())
    segs = whisper_resp["segments"]
    total_time = segs[-1]["end"] - segs[0]["start"]
    wpm = (total_words / total_time) * 60 if total_time > 0 else 0
    result["wpm"] = round(wpm, 2)

    # -------------------------
    # 2) 停頓比例
    # -------------------------
    silences = []
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i-1]["end"]
        if gap > 0.25:
            silences.append(gap)

    total_silence = sum(silences)
    result["silence_ratio"] = round(total_silence / total_time, 3)

    # -------------------------
    # 3) 音量穩定度（Volume Stability）
    # -------------------------
    y, sr = sf.read(io.BytesIO(audio_bytes))
    frame_energy = librosa.feature.rms(y=y)[0]

    vol_mean = np.mean(frame_energy)
    vol_std = np.std(frame_energy)

    stability = 1 - (vol_std / (vol_mean + 1e-9))
    result["volume_stability"] = round(float(stability), 3)

    # -------------------------
    # 4) 填充詞比例
    # -------------------------
    filler_count = sum(whisper_resp["text"].count(f) for f in FILLERS)
    filler_ratio = filler_count / max(total_words, 1)
    result["filler_ratio"] = round(filler_ratio, 3)

    return result


# ============================================================
# ------------- RAG 知識庫載入（電資學生專用） ---------------
# ============================================================

class SimpleRAG:
    def __init__(self, folder="knowledge"):
        self.docs = []
        if not os.path.isdir(folder):
            return
        for fname in os.listdir(folder):
            if fname.endswith((".md", ".txt")):
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    self.docs.append((fname, f.read()))

    def retrieve(self, job, query, top_k=3):
        if not self.docs:
            return []
        q = query.lower()
        scored = []
        for name, text in self.docs:
            score = sum(q.count(tok)
                        for tok in q.split() if tok in text.lower())
            scored.append((score, text))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [x[1] for x in scored[:top_k] if x[0] > 0]


@st.cache_resource
def load_rag():
    return SimpleRAG("knowledge")


rag = load_rag()

# ============================================================
# -------------------- UI & Session 初始化 -------------------
# ============================================================

st.set_page_config(page_title="AI 虛擬面試官", page_icon="🧑‍🏫", layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("static/style.css")

st.title("🧑‍🏫 AI 虛擬面試官")

# Custom CSS for new components
st.markdown("""
<style>
    /* Additional custom styles can be added here */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
    }
</style>""", unsafe_allow_html=True)


def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


init_state("messages", [])
init_state("started", False)
init_state("resume_info", None)
init_state("candidate_id", "")
init_state("qa_list", [])
init_state("last_question", None)
init_state("grade_result", None)
init_state("selected_history_interview_id", None)
init_state("voice_mode", True)
init_state("play_tts_first_question", False)
init_state("last_speech_features", None)
init_state("last_audio_hash", None)
init_state("start_time", None)
init_state("time_limit_minutes", 10)
init_state("auto_end_reason", None)
init_state("etiquette_strikes", 0)
init_state("qualified_streak", 0)

# ============================================================
# PART 2 — 履歷解析、Prompt 生成、RAG、LLM 回覆
# ============================================================

# ------------------------------------------------------------
# Sidebar 設置
# ------------------------------------------------------------
with st.sidebar:
    st.header("面試設定")

    # 受試者 ID
    candidate_id = st.text_input(
        "受試者 ID（姓名 / 學號）", value=st.session_state.candidate_id)
    st.session_state.candidate_id = candidate_id

    if candidate_id:
        save_candidate(candidate_id)

    job_role = st.selectbox(
        "應徵職缺",
        [
            "後端工程師",
            "AI 工程師",
            "資料工程師",
            "前端工程師",
            "韌體工程師",
            "硬體工程師",
            "FPGA 工程師",
            "射頻工程師",
            "電力電子工程師",
            "嵌入式系統工程師",
        ]
    )
    st.session_state.job_role = job_role

    interview_style = st.selectbox(
        "面試風格",
        ["普通", "嚴格", "溫和"]
    )
    st.session_state.interview_style = interview_style

    st.markdown("---")
    st.subheader("履歷上傳（PDF）")
    uploaded_resume = st.file_uploader("選擇 PDF 履歷", type=["pdf"])

    st.caption("提示：可將公司/產品知識以 .md 或 .txt 放入 knowledge 資料夾，系統將自動載入。")

    st.markdown("---")
    st.subheader("自動結束設定")
    time_limit = st.slider(
        "超過此分鐘數自動結束面試",
        min_value=5,
        max_value=30,
        value=st.session_state.time_limit_minutes,
        step=1
    )
    st.session_state.time_limit_minutes = time_limit

    st.markdown("---")
    st.subheader("歷史紀錄")

    history = []
    if candidate_id:
        history = get_interviews(candidate_id)

    if history:
        options = [
            f"{h['timestamp']}｜{h['job_role']}｜ID:{h['interview_id']}"
            for h in history
        ]
        picked = st.selectbox("選擇一筆歷史紀錄：", options)
        idx = options.index(picked)
        st.session_state.selected_history_interview_id = history[idx]["interview_id"]
    else:
        st.caption("尚無歷史紀錄")

    st.markdown("---")
    if st.button("🔁 重置面試"):
        for key in [
            "messages", "started", "resume_info", "qa_list",
            "last_question", "grade_result", "last_speech_features",
            "last_audio_hash", "start_time", "auto_end_reason",
            "etiquette_strikes", "qualified_streak"
        ]:
            st.session_state[key] = None if key == "resume_info" else []
        st.session_state.started = False
        st.rerun()


# ------------------------------------------------------------
# 履歷解析（PDF → JSON）
# ------------------------------------------------------------
if uploaded_resume and st.session_state.resume_info is None:
    with st.spinner("AI 正在解析你的履歷…"):
        st.session_state.resume_info = parse_resume(uploaded_resume)
    st.success("履歷解析完成！")

# 展示履歷解析內容
with st.expander("📄 履歷解析結果"):
    ri = st.session_state.resume_info
    if ri:
        st.markdown("### 🧩 技能")
        st.write(", ".join(ri.get("skills", [])) or "（無）")

        st.markdown("### 📚 專案")
        for p in ri.get("projects", []):
            st.markdown(f"**{p['title']}** — {p['description']}")
            st.caption("技術：" + ", ".join(p.get("tech_stack", [])))

        st.markdown("### 💼 工作經驗")
        for w in ri.get("work_experience", []):
            st.markdown(
                f"**{w['company']} / {w['position']} ({w['duration']})**")
            st.write(w["description"])

        st.markdown("### 🎓 學歷")
        for e in ri.get("education", []):
            st.markdown(f"- {e['school']} — {e['degree']} ({e['duration']})")

        st.markdown("### 📝 自我摘要")
        st.write(ri.get("summary", "（無）"))
    else:
        st.caption("尚未上傳履歷。")


# ------------------------------------------------------------
# Prompt 建構器（含 RAG）
# ------------------------------------------------------------
def build_system_prompt(job, style, resume_info=None, rag_snippets=None):

    style_desc = {
        "普通": "語氣專業，提問自然。",
        "嚴格": "語氣直接、追問細節、有壓力感。",
        "溫和": "語氣親切、鼓勵式提問。",
    }[style]

    ROLE_COMPETENCIES = {
        "後端工程師": [
            "系統設計與架構取捨（可用性、延展性、安全）",
            "資料庫與快取（SQL/NoSQL/索引/交易）",
            "API 設計與效能（REST/GraphQL、觀測性、CI/CD）",
            "併發與可靠性（鎖、重試、排程、併發模型）",
        ],
        "AI 工程師": [
            "模型選型與微調（LLM、Transformer、向量索引）",
            "RAG 與資料管線（檢索、Chunk、Embedding、評估）",
            "部署與效能（批次、量化、快取、可觀測性）",
            "資料與安全（隱私、漂移監控、資料品質）",
        ],
        "資料工程師": [
            "資料管線設計（批/流、重試、回填）",
            "儲存與模型（湖倉、分割、索引、格式）",
            "調度與治理（Airflow/工作流、品質、血緣、成本控管）",
            "可用性與擴展（分散式處理、彈性、監控）",
        ],
        "前端工程師": [
            "架構與狀態管理（React/Vue、路由、快取策略）",
            "性能優化（首屏/包體、Lazy load、SSR/CSR）",
            "可用性與無障礙（a11y、設計系統、一致性）",
            "前後端協作與測試（API 對齊、E2E/單元測試、CI）",
        ],
        "韌體工程師": [
            "MCU/SoC 架構與驅動開發（I2C/SPI/UART/USB）",
            "RTOS/裸機設計（排程、中斷、低功耗）",
            "韌體測試與量產（DFU/OTA、量測、自動化測試）",
            "效能與可靠度（記憶體/功耗優化、除錯與追蹤）",
        ],
        "硬體工程師": [
            "電路設計與佈局（原理圖、PCB、SI/PI）",
            "元件選型與可靠度（Derating、EMI/EMC、ESD）",
            "量測驗證（示波器、頻譜分析、ATE）",
            "量產導入（DFM/DFA、BOM 成本、良率改善）",
        ],
        "FPGA 工程師": [
            "RTL/HDL 設計（Verilog/VHDL）、時序約束（SDC）",
            "高速介面與 IP（PCIe、Ethernet、DDR、SerDes）",
            "驗證與除錯（仿真、邏輯分析、LA/ILA）",
            "資源/功耗/時序優化（P&R、floorplanning）",
        ],
        "射頻工程師": [
            "RF 前端設計（PA/LNA/Filter、匹配、天線）",
            "量測與調校（VNA、頻譜、諧波、隔離度）",
            "EMI/EMC 與法規（認證流程、整改方案）",
            "系統整合（RF + Baseband、干擾分析、熱管理）",
        ],
        "電力電子工程師": [
            "電源拓撲與控制（Buck/Boost、PFC、LLC）",
            "磁性元件與散熱（變壓器/電感設計、熱阻分析）",
            "保護與可靠度（OCP/OVP/OTP、安規與認證）",
            "效率與 EMI 最佳化（佈局、補償、開關損耗）",
        ],
        "嵌入式系統工程師": [
            "系統整合（Sensor/Actuator、通訊匯流排）",
            "作業系統與驅動（Linux/RTOS、Device Tree、驅動模型）",
            "效能與功耗（CPU/GPU/加速器、DVFS、低功耗模式）",
            "安全與更新（Secure Boot、OTA、故障復原）",
        ],
    }

    # ===== 履歷內容 =====
    resume_context = ""
    if resume_info:
        skills = resume_info.get("skills", [])
        resume_context += f"候選人技能：{', '.join(skills)}\n" if skills else ""

        if resume_info.get("projects"):
            resume_context += "專案：\n"
            for p in resume_info["projects"]:
                resume_context += f"- {p['title']}: {p['description']}\n"

    # ===== RAG =====
    rag_context = ""
    if rag_snippets:
        rag_context += "\n以下為職缺相關的技術知識片段（RAG）：\n"
        for i, sn in enumerate(rag_snippets, 1):
            rag_context += f"[{i}] {sn}\n"

    competencies = ROLE_COMPETENCIES.get(job, [])
    comp_text = "\n".join(
        f"- {c}" for c in competencies) if competencies else ""

    return f"""
你是一位專業的 **{job}** 面試官。

面試風格：{style_desc}

該職缺核心能力：
{comp_text}

請遵守規則：
1. 每次只問一題。
2. 問題需有技術深度，聚焦職缺能力。
3. 若候選人答不完整，追問更細。
4. 用繁體中文。

候選人資訊：
{resume_context}

技術知識（RAG）：
{rag_context}

開始面試，請提出第一題：自我介紹。
""".strip()


# ------------------------------------------------------------
# LLM Response (with RAG query)
# ------------------------------------------------------------
def call_llm(job, style, history, resume_info=None):

    # ---- RAG 查詢字串 ----
    query_parts = [f"職缺：{job}"]

    last_q = None
    last_a = None

    for role, msg in reversed(history):
        if role == "assistant" and last_q is None:
            last_q = msg
        elif role == "user" and last_a is None:
            last_a = msg
        if last_q and last_a:
            break

    if last_q:
        query_parts.append("上一題：" + last_q[:80])
    if last_a:
        query_parts.append("上一答：" + last_a[:80])

    if resume_info and resume_info.get("skills"):
        query_parts.append("技能：" + ", ".join(resume_info["skills"]))

    rag_query = "；".join(query_parts)

    # ---- 根據職缺自動排序 RAG ----
    role_pref = {
        "後端工程師": ["algorithms", "datastructures", "system_design", "database"],
        "AI 工程師": ["ai_ml", "algorithms", "computer_arch"],
        "資料工程師": ["database", "system_design"],
        "前端工程師": ["algorithms", "system_design"],
        "韌體工程師": ["firmware", "rtos", "driver", "embedded"],
        "硬體工程師": ["pcb", "emi", "layout", "analog"],
        "FPGA 工程師": ["fpga", "rtl", "vhdl", "verilog", "timing"],
        "射頻工程師": ["rf", "antenna", "emi", "emc"],
        "電力電子工程師": ["pfc", "power", "thermal", "converter"],
        "嵌入式系統工程師": ["embedded", "linux", "driver", "device tree"],
    }.get(job, [])

    raw_snippets = rag.retrieve(job, rag_query, top_k=5)
    rag_snippets = sorted(
        raw_snippets,
        key=lambda x: any(tag in x.lower() for tag in role_pref),
        reverse=True
    )[:3]

    # ---- System prompt ----
    system_prompt = build_system_prompt(
        job,
        style,
        resume_info=resume_info,
        rag_snippets=rag_snippets
    )

    # ---- Messages ----
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in history:
        messages.append({"role": role, "content": content})

    # ---- 呼叫 OpenAI ----
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return resp.choices[0].message.content


def parse_json_response(text: str):
    """Parse model output as JSON; try extracting the first {...} block on failure."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                return None
    return None


def evaluate_auto_end(qa_list, job_role, resume_info=None):
    """
    Use LLM to decide whether to auto-end the interview after the latest answer.
    Returns dict with keys: label, reason, action (end | warn | continue).
    """
    if not qa_list:
        return None

    last_qa = qa_list[-1]
    skills = ", ".join(resume_info.get("skills", [])) if resume_info else ""
    prompt = f"""
你是面試官的助手，請判斷是否要結束面試。請偏向「繼續」並給予提醒，只有在明顯且嚴重的情況才結束。根據最新回答，判斷：
- 是否嚴重違反面試禮儀（持續攻擊或無禮）。輕微失禮請用 warn，不要直接結束。
- 是否明顯不符合職缺要求（多次回答與職缺無關且無改善跡象）。
- 是否已經足夠確認其合格（能更深入追問）

請以 JSON 輸出：
{{
  "action": "end" | "warn" | "continue",
  "label": "etiquette" | "unwilling" | "not_qualified" | "qualified" | "continue",
  "reason": "簡短中文理由"
}}

職缺：{job_role}
履歷技能：{skills}
最新提問：{last_qa["question"]}
候選人回答：{last_qa["answer"]}
累計題數：{len(qa_list)}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    data = parse_json_response(resp.choices[0].message.content)
    if data:
        return {
            "label": data.get("label", "continue"),
            "action": data.get("action", "continue"),
            "reason": data.get("reason", "系統判定")
        }
    return None


def check_time_limit():
    """Return reason string if time limit exceeded; otherwise None."""
    limit = st.session_state.get("time_limit_minutes")
    start = st.session_state.get("start_time")
    if start and limit:
        elapsed = (datetime.now().timestamp() - start) / 60
        if elapsed >= limit:
            return f"已超過設定的 {int(limit)} 分鐘，系統自動結束。"
    return None


# ============================================================
# 報告強化：推薦等級 / 關鍵字命中 / 簡報內容
# ============================================================


def compute_recommendation(overall_scores):
    # 安全轉換為數字，避免 LLM 回傳文字造成錯誤
    vals = []
    for v in overall_scores.values():
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return "On Hold", 0

    avg = sum(vals) / len(vals)
    match_score = round(avg / 5 * 100, 1)
    if avg >= 4.5:
        rec = "Strong Hire"
    elif avg >= 4.0:
        rec = "Hire"
    elif avg >= 3.0:
        rec = "On Hold"
    else:
        rec = "Reject"
    return rec, match_score


ROLE_KEYWORDS = {
    "後端工程師": ["microservices", "REST", "graphql", "ci/cd", "kubernetes", "docker", "database", "redis", "rabbitmq"],
    "AI 工程師": ["transformer", "llm", "rag", "fine-tune", "pytorch", "tensorflow", "mlops", "vector db"],
    "資料工程師": ["etl", "spark", "hadoop", "kafka", "data pipeline", "warehouse", "airflow"],
    "前端工程師": ["react", "vue", "typescript", "webpack", "vite", "ui/ux", "accessibility"],
    "韌體工程師": ["firmware", "rtos", "i2c", "spi", "uart", "ota", "bootloader", "low power"],
    "硬體工程師": ["pcb", "s-parameters", "emi", "emc", "esd", "dfm", "dfa", "power integrity", "signal integrity"],
    "FPGA 工程師": ["fpga", "rtl", "verilog", "vhdl", "timing", "sdc", "pcie", "ddr", "serdes"],
    "射頻工程師": ["rf", "lna", "pa", "antenna", "vna", "s11", "emi", "emc"],
    "電力電子工程師": ["buck", "boost", "pfc", "llc", "switching", "emi", "transformer", "inductor", "thermal"],
    "嵌入式系統工程師": ["embedded", "linux", "rtos", "device tree", "driver", "spi", "i2c", "can", "ota"],
}


def compute_keyword_hits(qa_list, role):
    keywords = ROLE_KEYWORDS.get(role, [])
    if not keywords or not qa_list:
        return 0, {}
    all_text = " ".join(x["answer"] for x in qa_list).lower()
    counts = {kw: all_text.count(kw) for kw in keywords}
    total = sum(counts.values())
    return total, counts


def extract_strengths_risks(overall_scores):
    num_scores = {}
    for k, v in overall_scores.items():
        try:
            num_scores[k] = float(v)
        except Exception:
            continue

    strengths = [k for k, v in num_scores.items() if v >= 4.2]
    risks = [k for k, v in num_scores.items() if v <= 3.0]
    name_map = {
        "technical": "技術深度",
        "communication": "表達清晰度",
        "structure": "結構化",
        "relevance": "相關性",
        "problem_solving": "解題力",
        "growth_potential": "成長潛力",
    }
    return [name_map.get(s, s) for s in strengths], [name_map.get(r, r) for r in risks]


def build_brief_eval(overall_summary, strengths, risks):
    parts = []
    parts.append(overall_summary.strip())
    if strengths:
        parts.append("亮點包括：" + "、".join(strengths) + "。")
    if risks:
        parts.append("需改善：" + "、".join(risks) + "。")
    txt = " ".join(parts)
    words = txt.split()
    if len(words) < 90:
        pad = " 本段由系統自動生成，概述候選人表現與風險，供 HR 快速參考。"
        txt += pad
    return txt


def build_improvement_tips(overall_scores, speech_features=None):
    tips = []

    def add(text):
        if text not in tips:
            tips.append(text)

    name_map = {
        "technical": "技術深度",
        "communication": "表達清晰度",
        "structure": "結構化",
        "relevance": "相關性",
        "problem_solving": "解題力",
        "growth_potential": "成長潛力",
    }

    for k, v in overall_scores.items():
        try:
            score = float(v)
        except Exception:
            continue
        label = name_map.get(k, k)
        if k == "technical" and score < 4:
            add(f"{label}：補強核心技術原理與案例細節，回答時加入架構/效能/安全的量化指標。")
        if k == "communication" and score < 4:
            add(f"{label}：精簡開場，先講結論再補充背景，避免冗長鋪陳。")
        if k == "structure" and score < 4:
            add(f"{label}：採用 STAR / PREP 結構拆解，先列步驟或要點再展開。")
        if k == "relevance" and score < 4:
            add(f"{label}：回扣職缺需求與情境，避免離題，結尾補一句與目標的連結。")
        if k == "problem_solving" and score < 4:
            add(f"{label}：說清楚假設、風險與權衡，描述你如何驗證或 rollback。")
        if k == "growth_potential" and score < 4:
            add(f"{label}：補充近期學習或 side project，展現自我驅動與迭代。")

    if speech_features:
        if speech_features.get("silence_ratio", 0) > 0.25:
            add("口語停頓偏多：提前列提綱、用短句回答，避免長時間空白。")
        if speech_features.get("filler_ratio", 0) > 0.05:
            add("填充詞偏多：練習停頓替代『嗯、呃』，用『讓我確認一下重點』過渡。")
        if speech_features.get("volume_stability", 1) < 0.6:
            add("音量穩定度不足：保持中低速、降低情緒波動，讓重點更清楚。")

    if not tips:
        tips.append("整體表現均衡，維持目前回答框架即可。")

    return tips


def is_brief_greeting(text: str) -> bool:
    """Detect very short greetings to avoid over-penalizing etiquette."""
    if not text:
        return False
    t = text.strip().lower()
    greetings = ["hi", "hello", "hey", "你好", "嗨", "您好"]
    if len(t) <= 12 and any(g in t for g in greetings):
        return True
    words = t.split()
    return len(words) <= 3 and any(g in t for g in greetings)


def end_interview(reason_label="manual", reason_detail=None):
    """Run grading once and store auto-end reason."""
    if st.session_state.get("grade_result"):
        return
    if not st.session_state.qa_list:
        st.warning("你尚未回答任何題目，無法進行評分。")
        return

    result = grade_interview(
        st.session_state.qa_list,
        st.session_state.job_role if "job_role" in st.session_state else None,
        st.session_state.resume_info,
        speech_features=st.session_state.last_speech_features
    )

    st.session_state.grade_result = result
    st.session_state.started = False
    st.session_state.auto_end_reason = reason_detail or reason_label

    if st.session_state.candidate_id:
        interview_id = save_interview(
            candidate_id=st.session_state.candidate_id,
            job_role=st.session_state.job_role if "job_role" in st.session_state else "",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary=result["overall"]["summary"],
        )

        for qa in st.session_state.qa_list:
            save_qa(interview_id, qa["question"], qa["answer"])

        save_scores(interview_id, result["overall"])


# ============================================================
# PART 3 — 面試流程（開始面試 + 語音回答 + TTS + Whisper）
# ============================================================

# ------------------------------------------------------------
# 顯示歷史對話訊息
# ------------------------------------------------------------
for role, content in st.session_state.messages:
    st.chat_message(role).markdown(content)


# ------------------------------------------------------------
# 尚未開始面試
# ------------------------------------------------------------
if not st.session_state.started:

    if st.button("▶️ 開始面試"):

        # 生成第一題（通常是自我介紹）
        first_reply = call_llm(
            job_role,
            interview_style,
            [],
            resume_info=st.session_state.resume_info
        )

        st.session_state.messages.append(("assistant", first_reply))
        st.session_state.last_question = first_reply
        st.session_state.started = True
        st.session_state.start_time = datetime.now().timestamp()
        st.session_state.grade_result = None
        st.session_state.auto_end_reason = None
        st.session_state.last_audio_hash = None
        st.session_state.etiquette_strikes = 0
        st.session_state.qualified_streak = 0

        # ⭐ 關鍵：第一題 TTS 必須延後一輪播放
        if st.session_state.voice_mode:
            st.session_state.play_tts_first_question = True

        st.rerun()


# ------------------------------------------------------------
# 第一題 TTS 播放（避免被 rerun 吃掉）
# ------------------------------------------------------------
if st.session_state.get("play_tts_first_question", False):
    st.session_state.play_tts_first_question = False   # 播一次就關掉

    text = st.session_state.last_question
    audio_bytes = synthesize_speech(text)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")


# ------------------------------------------------------------
# 面試已經開始 → 使用者回答（語音 / 文字）
# ------------------------------------------------------------
if st.session_state.started:

    # 時間到自動結束（有作答才評分）
    if not st.session_state.get("grade_result"):
        time_reason = check_time_limit()
        if time_reason and st.session_state.qa_list:
            end_interview("time_limit", time_reason)
            st.info(time_reason)
            st.rerun()

    st.markdown("### 🧑‍💬 請回答：")

    # 語音錄製與文字輸入上下排列，維持在同一區塊
    voice_answer = None

    st.markdown("#### 🎤 語音錄音")
    audio_rec = st.audio_input(
        "點擊錄音開始作答",
        label_visibility="collapsed",
    )

    st.markdown("#### 📝 文字回答")
    text_answer = st.chat_input("請輸入你的回答…", key="text_answer")

    if audio_rec:
        audio_bytes = audio_rec.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if audio_hash != st.session_state.last_audio_hash:
            with st.spinner("Whisper 正在辨識語音…"):
                whisper_resp = speech_to_text(audio_rec)

            voice_answer = whisper_resp["text"]

            # ===== 語音特徵分析 =====
            speech_features = analyze_speech_features(
                whisper_resp, audio_bytes)
            st.session_state.last_speech_features = speech_features
            st.session_state.last_audio_hash = audio_hash

            st.success("語音辨識完成！")
        else:
            st.info("這段錄音已處理過，未重複送出。")

    user_input = voice_answer if voice_answer else text_answer

    if user_input:

        # --------- 記錄上一題+使用者回答（QA） -----------
        st.session_state.qa_list.append({
            "question": st.session_state.last_question,
            "answer": user_input
        })

        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        # --------- 檢查是否自動結束 ----------
        proceed_with_question = True
        decision = evaluate_auto_end(
            st.session_state.qa_list,
            job_role,
            st.session_state.resume_info
        )
        # time limit is handled above; here handle etiquette/fit logic
        if decision:
            label = decision.get("label")
            action = decision.get("action")
            reason = decision.get("reason", "")

            if label == "etiquette":
                # 每次違反禮儀都累計，滿 3 次自動結束
                st.session_state.etiquette_strikes += 1
                strikes = st.session_state.etiquette_strikes
                if strikes >= 3 and action == "end":
                    end_interview(
                        "etiquette", f"第 {strikes} 次違反禮儀：{reason or '多次簡短/不禮貌回覆'}")
                    st.info(f"面試已自動結束：{reason or '多次違反禮儀'}")
                    st.rerun()
                else:
                    st.warning(
                        f"注意面試禮儀（{strikes}/3）：{reason or '請提供完整、自信的回覆。'}")
                proceed_with_question = False  # 讓使用者重新作答，不要中斷 app

            elif label == "qualified":
                st.session_state.qualified_streak += 1
                if st.session_state.qualified_streak >= 10:
                    end_interview("offer", "連續 10 題符合標準，恭喜獲得錄用！")
                    st.success("恭喜！你已獲得錄用，面試結束。")
                    st.rerun()
                # continue asking deeper questions automatically

            elif action == "end" and label in ("unwilling", "not_qualified"):
                end_interview(label, reason)
                st.info(f"面試已自動結束：{reason}")
                st.rerun()

        if proceed_with_question:
            # --------- 呼叫面試官取得下一題 ----------
            assistant_reply = call_llm(
                job_role,
                interview_style,
                st.session_state.messages,
                resume_info=st.session_state.resume_info,
            )

            st.session_state.messages.append(("assistant", assistant_reply))
            st.chat_message("assistant").markdown(assistant_reply)
            st.session_state.last_question = assistant_reply

            # --------- TTS 播放下一題 ----------
            if st.session_state.voice_mode:
                tts_audio = synthesize_speech(assistant_reply)
                if tts_audio:
                    st.audio(tts_audio, format="audio/mp3")

# ============================================================
# PART 4 — AI 面試評分（含語音特徵 + 語音建議）
# ============================================================

# ------------------------------------------------------------
# 評分按鈕
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 面試評分（AI 分析）")

if st.button("📊 結束面試並進行 AI 評分"):

    if st.session_state.grade_result:
        st.info("本次面試已完成評分。")
    else:
        with st.spinner("AI 正在分析你的整場面試……"):
            end_interview("manual", "手動結束面試")

        st.success("評分完成！向下捲動查看分析結果。")


# ------------------------------------------------------------
# 顯示評分結果
# ------------------------------------------------------------
if (
    not st.session_state.grade_result
    and st.session_state.qa_list
    and not st.session_state.started
):
    # 若已停止面試但尚未寫入評分，補算一次
    end_interview("manual", "系統自動補算評分")

if st.session_state.grade_result:

    result = st.session_state.grade_result
    overall = result["overall"]
    per_question = result["per_question"]

    tech = overall["technical"]
    comm = overall["communication"]
    struct = overall["structure"]
    rel = overall["relevance"]
    ps = overall["problem_solving"]
    gp = overall["growth_potential"]

    sf = st.session_state.last_speech_features
    from grader import generate_speech_feedback

    # Report metadata derived from scores/keywords
    rec, match_score = compute_recommendation(overall)
    _, kw_detail = compute_keyword_hits(st.session_state.qa_list, job_role)
    strengths, risks = extract_strengths_risks(overall)
    brief_eval = build_brief_eval(overall["summary"], strengths, risks)
    improvement_tips = build_improvement_tips(overall, sf)

    def speech_brief(features):
        if not features:
            return "未提供語音，無法進行口語分析。"
        wpm = features.get("wpm", 0)
        silence = features.get("silence_ratio", 0)
        filler = features.get("filler_ratio", 0)
        stability = features.get("volume_stability", 0)
        parts = []
        # 流暢度
        if 80 <= wpm <= 180:
            parts.append("語速在可理解範圍，流暢度良好")
        elif wpm < 80:
            parts.append("語速偏慢，需加快節奏避免冗長")
        else:
            parts.append("語速偏快，建議放慢以便理解")
        # 停頓與贅詞
        if silence > 0.25:
            parts.append("停頓比例偏高，建議先組織再回答")
        if filler > 0.05:
            parts.append("填充詞較多，影響專業感")
        # 音量穩定度
        if stability < 0.6:
            parts.append("音量起伏較大，需提升穩定度")
        else:
            parts.append("音量穩定度尚可")
        return "；".join(parts)

    st.markdown("## 📊 面試分析結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("匹配分數", f"{match_score}/100")
    c2.metric("AI 建議", rec)
    c3.metric("題目數", len(per_question))
    c4.metric("語音樣本", "有" if sf else "無")

    def build_report_md():
        lines = []
        lines.append("# AI 面試評估報告")
        lines.append(f"- 報告生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"- 候選人：{st.session_state.candidate_id}")
        lines.append(f"- 應徵職位：{job_role}")
        if st.session_state.auto_end_reason:
            lines.append(f"- 結束原因：{st.session_state.auto_end_reason}")
        lines.append("")

        # 1. 摘要總覽（全中文）
        lines.append("## 1. 摘要總覽")
        lines.append(f"- AI 總體建議：{rec}")
        lines.append(f"- 整體匹配分數：{match_score}/100")
        lines.append(f"- 主要優勢：{', '.join(strengths) if strengths else '無明顯優勢'}")
        lines.append(f"- 潛在風險：{', '.join(risks) if risks else '無明顯風險'}")
        lines.append(f"- AI 簡述：{brief_eval}")
        lines.append("")

        # 2. 技術能力（以現有訊號做 RAG Proxy）
        kw_hits_total = sum(kw_detail.values()) if kw_detail else 0
        lines.append("## 2. 技術能力評估（RAG 近似）")
        lines.append(f"- 準確度（以 Relevance 代表）：{rel}/5")
        lines.append(f"- 深度（以 Technical 代表）：{tech}/5")
        lines.append(f"- 結構化表達（Structure）：{struct}/5")
        lines.append(f"- 關鍵字命中數：{kw_hits_total}")
        if kw_detail and any(v > 0 for v in kw_detail.values()):
            lines.append("- 關鍵字明細：")
            for k, v in sorted(kw_detail.items(), key=lambda kv: kv[1], reverse=True):
                if v > 0:
                    lines.append(f"  - {k}：{v}")
        else:
            lines.append("- 關鍵字明細：尚未偵測到命中")
        lines.append(f"- 面試答題樣本數：{len(per_question)}")
        lines.append("")

        # 3. 軟實力 / 行為表現（語音特徵）
        lines.append("## 3. 軟實力與行為表現（語音）")
        lines.append(f"- 溝通表現（整體）：{comm}/5")
        if sf:
            lines.append(f"- 語速 WPM：{sf['wpm']}")
            lines.append(f"- 靜音比例：{sf['silence_ratio']}")
            lines.append(f"- 贅詞比例：{sf['filler_ratio']}")
            lines.append(f"- 音量穩定度：{sf['volume_stability']}")
            lines.append(f"- 口語表現總結：{speech_brief(sf)}")
            lines.append("- 情緒與態度：目前未啟用情緒/表情分析（暫不提供）")
            lines.append("- 語音建議（AI）：")
            lines.append(generate_speech_feedback(sf))
        else:
            lines.append("- 語音特徵：未提供語音，無法分析")
            lines.append("- 情緒與態度：未偵測")
            lines.append(f"- 口語表現總結：{speech_brief(sf)}")
        lines.append("")

        # 4. 改進建議
        lines.append("## 4. 改進建議（AI）")
        for tip in improvement_tips:
            lines.append(f"- {tip}")
        lines.append("")

        # 5. 題目逐題分析
        lines.append("## 5. 題目逐題分析 Question-by-Question")
        for i, item in enumerate(per_question, 1):
            s = item['score']
            lines.append(f"### 題目 {i}")
            lines.append(f"- 問題：{item['question']}")
            lines.append(f"- 回答：{item['answer']}")
            lines.append(
                f"- 評分 Technical {s['technical']}/5，"
                f"Communication {s['communication']}/5，"
                f"Structure {s['structure']}/5，"
                f"Relevance {s['relevance']}/5，"
                f"Problem Solving {s['problem_solving']}/5，"
                f"Growth Potential {s['growth_potential']}/5"
            )
            lines.append(f"- AI 回饋：{item['feedback']}")
            lines.append("")

        # 6. 附錄 / 原始資料
        lines.append("## 6. 附錄與原始分數")
        lines.append(f"- Overall Technical：{tech}")
        lines.append(f"- Overall Communication：{comm}")
        lines.append(f"- Overall Structure：{struct}")
        lines.append(f"- Overall Relevance：{rel}")
        lines.append(f"- Overall Problem Solving：{ps}")
        lines.append(f"- Overall Growth Potential：{gp}")
        lines.append(f"- 系統摘要：{overall['summary']}")

        return "\n".join(lines)

    report_md = build_report_md()

    st.markdown(report_md)

    st.subheader("🎙️ 語音分析")
    if sf:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("語速 WPM", sf.get("wpm"))
        c2.metric("靜音比例", sf.get("silence_ratio"))
        c3.metric("贅詞比例", sf.get("filler_ratio"))
        c4.metric("音量穩定度", sf.get("volume_stability"))
        st.markdown(f"- 口語表現總結：{speech_brief(sf)}")
        st.markdown("**AI 語音建議**")
        st.markdown(generate_speech_feedback(sf))
    else:
        st.info("本次未提供語音，無法產生語音報告。請錄製語音作答以獲得口語分析。")

    import tempfile
    image_paths = []

    categories = ["technical", "communication", "structure",
                  "relevance", "problem_solving", "growth_potential"]
    labels_zh = ["技術", "表達", "結構", "相關", "解題", "成長"]
    scores = [tech, comm, struct, rel, ps, gp]
    values = scores + scores[:1]
    angles = np.linspace(0, 2*np.pi, len(categories) + 1)

    fig_dl, ax_dl = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax_dl.plot(angles, values, linewidth=2)
    ax_dl.fill(angles, values, alpha=0.25)
    ax_dl.set_thetagrids(angles[:-1] * 180/np.pi, labels_zh)
    ax_dl.set_ylim(0, 5)
    ax_dl.set_yticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    st.pyplot(fig_dl)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
        fig_dl.savefig(img_tmp.name, bbox_inches="tight")
        image_paths.append(img_tmp.name)
    plt.close(fig_dl)

    if st.session_state.selected_history_interview_id:
        ref_scores = get_scores(st.session_state.selected_history_interview_id)
        if ref_scores:
            ref_vals = [
                ref_scores['technical'],
                ref_scores['communication'],
                ref_scores['structure'],
                ref_scores['relevance'],
                ref_scores['problem_solving'],
                ref_scores['growth_potential'],
            ]
            ref_plot = ref_vals + ref_vals[:1]
            cur_plot = values

            fig_cmp, ax_cmp = plt.subplots(
                figsize=(6, 6), subplot_kw={"polar": True})
            ax_cmp.plot(angles, ref_plot, "r--", linewidth=1.8, label="歷史紀錄")
            ax_cmp.plot(angles, cur_plot, "b-", linewidth=2.2, label="本次結果")
            ax_cmp.fill(angles, cur_plot, alpha=0.25)
            ax_cmp.set_thetagrids(angles[:-1] * 180/np.pi, labels_zh)
            ax_cmp.set_ylim(0, 5)
            ax_cmp.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
            plt.tight_layout()
            st.pyplot(fig_cmp)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp2:
                fig_cmp.savefig(img_tmp2.name, bbox_inches="tight")
                image_paths.append(img_tmp2.name)
            plt.close(fig_cmp)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        export_pdf(tmp.name, report_md, image_paths=image_paths)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()

    html_content = export_html(report_md)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇️ 下載 Markdown 報告",
            data=report_md,
            file_name="interview_report.md",
            mime="text/markdown",
        )
    with dl2:
        st.download_button(
            "⬇️ 下載 PDF 報告",
            data=pdf_bytes,
            file_name="interview_report.pdf",
            mime="application/pdf",
        )
    with dl3:
        st.download_button(
            "⬇️ 下載 HTML 報告",
            data=html_content,
            file_name="interview_report.html",
            mime="text/html",
        )
