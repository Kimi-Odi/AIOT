# ================================================================
# grader.py — AI 虛擬面試官評分模組（含語音特徵調整 + 語音改善建議）
# ================================================================

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================================================
# 🔹 語音特徵調整（B 功能）
# ================================================================


def speech_feature_adjustment(features):
    if not features:
        return 1.0

    wpm = features["wpm"]
    silence = features["silence_ratio"]
    stability = features["volume_stability"]
    filler = features["filler_ratio"]

    # 語速
    if wpm < 80:
        wpm_score = 0.7
    elif 80 <= wpm <= 180:
        wpm_score = 1.0
    else:
        wpm_score = 0.8

    # 停頓
    if silence < 0.1:
        silence_score = 1.0
    elif silence < 0.25:
        silence_score = 0.85
    else:
        silence_score = 0.65

    # 音量穩定度
    stability_score = min(max(stability, 0), 1)

    # 填充詞
    if filler < 0.02:
        filler_score = 1.0
    elif filler < 0.05:
        filler_score = 0.8
    else:
        filler_score = 0.65

    final = (wpm_score + silence_score + stability_score + filler_score) / 4
    return round(final, 3)

# ================================================================
# 🔹 語音改善建議（D 功能）
# ================================================================


def generate_speech_feedback(features):
    if not features:
        return "本次未提供語音回答，因此無法產生語音表達建議。"

    wpm = features["wpm"]
    silence = features["silence_ratio"]
    stability = features["volume_stability"]
    filler = features["filler_ratio"]

    fb = []

    if wpm < 100:
        fb.append(f"- 語速 {wpm} WPM：偏慢，可提升流暢度。")
    elif wpm > 180:
        fb.append(f"- 語速 {wpm} WPM：偏快，建議放慢語句。")
    else:
        fb.append(f"- 語速 {wpm} WPM：表現良好。")

    if silence > 0.25:
        fb.append(f"- 停頓比例 {silence}：停頓偏多，可先組織句子再回答。")
    else:
        fb.append(f"- 停頓比例 {silence}：自然。")

    if stability < 0.6:
        fb.append(f"- 音量穩定度 {stability}：音量起伏明顯，可加強穩定。")
    else:
        fb.append(f"- 音量穩定度 {stability}：良好。")

    if filler > 0.05:
        fb.append(f"- 填充詞比例 {filler}：口頭禪偏多，可練習避免。")
    else:
        fb.append(f"- 填充詞比例 {filler}：正常。")

    fb.append("\n建議每天錄音練習 3~5 分鐘，會明顯改善口語表達。")

    return "\n".join(fb)

# ================================================================
# 🔹 逐題評分
# ================================================================


def grade_single_qa(question, answer, speech_features=None):
    def parse_json_response(text):
        """
        Try to parse the model output as JSON; fall back to extracting the first
        {...} block when strict parsing fails.
        """
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    prompt = f"""
你是一位專業面試官，請針對候選人的回答進行逐題評分（1~5 分，可含小數）。

題目：{question}
回答：{answer}

請以 JSON 回傳：
{{
  "technical": x,
  "communication": x,
  "structure": x,
  "relevance": x,
  "problem_solving": x,
  "growth_potential": x,
  "feedback": "一句話回饋"
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = resp.choices[0].message.content
    data = parse_json_response(content)
    if not data:
        # Provide safe defaults so grading doesn't crash if JSON parsing fails.
        data = {
            "technical": 3,
            "communication": 3,
            "structure": 3,
            "relevance": 3,
            "problem_solving": 3,
            "growth_potential": 3,
            "feedback": content.strip() if content else "（模型回覆無法解析為 JSON）",
        }
    # clamp and format scores to 1~5 with decimals
    for k in ["technical", "communication", "structure", "relevance", "problem_solving", "growth_potential"]:
        try:
            data[k] = max(1, min(5, float(data[k])))
        except Exception:
            data[k] = 1.0

    # ⭐ 將語音特徵加權
    if speech_features:
        factor = speech_feature_adjustment(speech_features)
        data["communication"] = round(data["communication"] * factor, 2)
        data["structure"] = round(data["structure"] * (0.7 + factor * 0.3), 2)

    return data

# ================================================================
# 🔹 整場面試評分
# ================================================================


def grade_interview(qa_list, job_role, resume_info=None, speech_features=None):

    per_question = []

    for qa in qa_list:
        score = grade_single_qa(
            qa["question"],
            qa["answer"],
            speech_features=speech_features
        )
        per_question.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "score": score,
            "feedback": score["feedback"]
        })

    n = len(per_question)
    overall = {
        "technical": 0,
        "communication": 0,
        "structure": 0,
        "relevance": 0,
        "problem_solving": 0,
        "growth_potential": 0,
    }

    for item in per_question:
        s = item["score"]
        for key in overall:
            overall[key] += s[key]

    for key in overall:
        overall[key] = round(overall[key] / n, 2)

    # 整體評論（LLM）
    summary_prompt = f"""
根據以下分數（1~5）撰寫一段 3~5 句的整體評論（繁體中文）：

職缺：{job_role}
分數：{overall}

請給流暢段落，不需列點。
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": summary_prompt}]
    )

    overall["summary"] = resp.choices[0].message.content.strip()

    return {
        "overall": overall,
        "per_question": per_question
    }
