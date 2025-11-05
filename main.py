# main.py
from typing import List, Dict, Optional
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# =========================================
# تحميل المتغيرات من .env (محلياً) + بيئة Render
# =========================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# =========================================
# بيانات الـ Benchmark (ممكن تطورها لاحقاً)
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
# نماذج الـ Pydantic
# =========================================
class Answer(BaseModel):
    text: str
    skill: str
    score: int  # 1–5


class AnalyzePayload(BaseModel):
    name: str
    track: str
    answers: List[Answer]

    # معلومات العميل (تجي من الـ pop-up)
    userId: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class AnalyzeResponse(BaseModel):
    overallScore: float
    skillScores: Dict[str, float]
    benchScores: Dict[str, int]
    aiReport: str


class SignupPayload(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None


class SignupResponse(BaseModel):
    userId: str


class AssessmentSummary(BaseModel):
    id: int
    created_at: datetime
    track: str
    overallScore: float
    skillScores: Dict[str, float]


class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str]
    assessments: List[AssessmentSummary]


# =========================================
# منطق التحليل + استدعاء GPT
# =========================================
def analyze_user(name: str, track: str, answers: List[Dict]) -> Dict:
    """
    answers: list of {"text": "..", "skill": "Technical", "score": 1..5}
    """
    # 1) تجميع الدرجات لكل skill
    buckets: Dict[str, List[int]] = {}
    for a in answers:
        skill = a["skill"]
        score = int(a["score"])
        buckets.setdefault(skill, []).append(score)

    # 2) حساب المتوسط وتحويله لـ 0–100
    skill_scores: Dict[str, float] = {}
    for skill, vals in buckets.items():
        avg = sum(vals) / len(vals)
        skill_scores[skill] = round((avg / 5.0) * 100.0, 1)

    overall = round(sum(skill_scores.values()) / len(skill_scores), 1) if skill_scores else 0.0
    bench_scores = BENCHMARKS.get(track, {})

    # 3) تجهيز برومبت GPT
    system_prompt = (
        "You are a senior career coach in the GCC market. "
        "Answer in modern Arabic with a Saudi/GCC tone. "
        "Be structured, clear, and practical."
    )

    payload = {
        "name": name,
        "track": track,
        "overall_score": overall,
        "skill_scores": skill_scores,
        "bench_scores": bench_scores,
    }

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
    version="1.1.0",
    description="Backend for Growday AI Skill Assessment + user profiles",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # بعدين حصرها على دومين الفرونت
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


# 1) /signup-lite  -> ينشئ/يحدّث مستخدم ويعيد userId
@app.post("/signup-lite", response_model=SignupResponse)
def signup_lite(payload: SignupPayload):
    # نبحث عن مستخدم بنفس الإيميل
    existing = (
        supabase.table("users")
        .select("id")
        .eq("email", payload.email)
        .maybe_single()
        .execute()
    )

    data = existing.data
    if data:
        user_id = data["id"]
        # نحدّث الاسم/الجوال لو تغيّروا
        supabase.table("users").update(
            {"name": payload.name, "phone": payload.phone}
        ).eq("id", user_id).execute()
        return {"userId": user_id}

    # لو ما حصلناه، ننشئ مستخدم جديد
    res = (
        supabase.table("users")
        .insert(
            {
                "name": payload.name,
                "email": payload.email,
                "phone": payload.phone,
            }
        )
        .execute()
    )

    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create user")

    user_id = res.data[0]["id"]
    return {"userId": user_id}


# 2) /analyze  -> تحليل + حفظ نتيجة الاختبار (لو فيه userId)
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(payload: AnalyzePayload):
    answers_list: List[Dict] = [a.dict() for a in payload.answers]
    result = analyze_user(
        name=payload.name,
        track=payload.track,
        answers=answers_list,
    )

    # لو فيه userId نحفظ الاختبار في جدول assessments
    if payload.userId:
        supabase.table("assessments").insert(
            {
                "user_id": payload.userId,
                "track": payload.track,
                "overall_score": result["overallScore"],
                "skill_scores": result["skillScores"],
            }
        ).execute()

    return result


# 3) /profile/{user_id}  -> يرجّع بروفايل العميل + قائمة اختبارات
@app.get("/profile/{user_id}", response_model=UserProfile)
async def get_profile(user_id: str):
    user_res = (
        supabase.table("users")
        .select("id,name,email,phone")
        .eq("id", user_id)
        .single()
        .execute()
    )
    user = user_res.data
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    assessments_res = (
        supabase.table("assessments")
        .select("id,track,overall_score,skill_scores,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    rows = assessments_res.data or []

    assessments: List[AssessmentSummary] = []
    for row in rows:
        created_raw = row["created_at"]
        # Supabase يعيد ISO string مثل "2025-11-05T21:00:00.000Z"
        created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        assessments.append(
            AssessmentSummary(
                id=row["id"],
                created_at=created_at,
                track=row["track"],
                overallScore=float(row["overall_score"]),
                skillScores=row["skill_scores"],
            )
        )

    return UserProfile(
        id=user["id"],
        name=user["name"],
        email=user["email"],
        phone=user.get("phone"),
        assessments=assessments,
    )


# للتشغيل المحلي: python main.py
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
