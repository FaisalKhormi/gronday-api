# main.py
from typing import List, Dict
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# =========================================
# إعداد OpenAI + .env
# =========================================
load_dotenv()  # يقرأ ملف .env لو موجود محلياً

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Set it in your cloud environment variables or .env file."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================
# بيانات المقارنة (Benchmark Scores)
# =========================================
BENCHMARKS: Dict[str, Dict[str, int]] = {
    "Data Analyst": {
        "Technical": 74,
        "Business": 70,
        "Communication": 68,
        "Problem Solving": 72,
    },
    "Product Manager": {
        "Business": 73,
        "Customer": 71,
        "Execution": 70,
        "Stakeholder": 69,
        "Leadership": 68,
    },
    "Software Engineer": {
        "Technical": 76,
        "Quality": 72,
        "Collaboration": 70,
        "Architecture": 69,
        "Growth": 71,
    },
}

# =========================================
# نماذج الطلب (Pydantic Models)
# =========================================
class Answer(BaseModel):
    text: str
    skill: str
    score: int  # 1–5

class AnalyzePayload(BaseModel):
    name: str
    track: str
    answers: List[Answer]

class AnalyzeResponse(BaseModel):
    overallScore: float
    skillScores: Dict[str, float]
    benchScores: Dict[str, int]
    aiReport: str


# =========================================
# منطق التحليل
# =========================================
def analyze_user(name: str, track: str, answers: List[Dict]) -> Dict:
    """
    answers: list of { "text": "...", "skill": "Technical", "score": 1..5 }
    """
    # 1) حساب درجات كل skill
    buckets: Dict[str, List[int]] = {}
    for a in answers:
        skill = a["skill"]
        score = int(a["score"])
        buckets.setdefault(skill, []).append(score)

    skill_scores: Dict[str, float] = {}
    for skill, vals in buckets.items():
        avg = sum(vals) / len(vals)
        # تحويل من 1–5 إلى 0–100
        skill_scores[skill] = round((avg / 5.0) * 100.0, 1)

    overall = (
        round(sum(skill_scores.values()) / len(skill_scores), 1)
        if skill_scores
        else 0.0
    )
    bench_scores = BENCHMARKS.get(track, {})

    # 2) استدعاء GPT لكتابة تقرير مهني
    payload = {
        "name": name,
        "track": track,
        "overall_score": overall,
        "skill_scores": skill_scores,
        "bench_scores": bench_scores,
    }

    system_prompt = (
        "You are a senior career coach in the GCC market. "
        "Answer in modern Arabic with a Saudi/GCC tone. "
        "Be structured, clear, and practical."
    )

    user_prompt = (
        "هذي نتيجة اختبار لمهارات مهنية (اسم المستخدم، المسار المهني، الدرجات):\n"
        f"{payload}\n\n"
        "اكتب تقرير مهني منظم يشمل:\n"
        "1) ملخص عام عن وضعه المهني.\n"
        "2) أهم نقاط القوة.\n"
        "3) أهم الفجوات اللي تمنعه من القفزة الجاية.\n"
        "4) مقارنة مختصرة بمستوى السوق (السعودية / الخليج).\n"
        "5) خطة عملية لمدة 30 يوم (خطوات أسبوعية واضحة).\n"
        "خليك صريح لكن مشجع، واستخدم أسلوب واقعي مو مبالغ فيه."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    ai_report = resp.choices[0].message.content or ""

    return {
        "overallScore": overall,
        "skillScores": skill_scores,
        "benchScores": bench_scores,
        "aiReport": ai_report,
    }


# =========================================
# إعداد FastAPI + CORS
# =========================================
app = FastAPI(
    title="Growday API",
    version="1.0.0",
    description="Backend for Growday AI Skill Assessment",
)

# في البداية نخليها مفتوحة للـ origins كلها، وبعد الإنتاج ممكن تضيقها على دومين فرونت معين
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # استبدلها لاحقاً بـ ["https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# Endpoints
# =========================================
@app.get("/")
def root():
    return {"status": "Growday API is running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(payload: AnalyzePayload):
    answers_list: List[Dict] = [a.dict() for a in payload.answers]
    result = analyze_user(
        name=payload.name,
        track=payload.track,
        answers=answers_list,
    )
    return result


# للتشغيل المحلي مباشرة: python main.py
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
