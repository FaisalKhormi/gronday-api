# main.py
from typing import List, Dict, Optional
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# =========================================
# إعداد .env + مفاتيح OpenAI
# =========================================
load_dotenv()  # يقرأ ملف .env محلياً (للتجربة)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Set it in your cloud environment variables or .env file."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================
# إعداد Supabase (الديتاوير هاوس)
# =========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "Supabase config missing: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.\n"
        "Set them as environment variables in Render and/or .env."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

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
# Pydantic Models
# =========================================
class Answer(BaseModel):
    text: str
    skill: str
    score: int  # 1–5


class AnalyzePayload(BaseModel):
    name: str
    track: str
    answers: List[Answer]

    # معلومات المستخدم (اختيارية، للربط مع البروفايل)
    userId: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None


class AnalyzeResponse(BaseModel):
    overallScore: float
    skillScores: Dict[str, float]
    benchScores: Dict[str, int]
    aiReport: str


class SignupLitePayload(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None


class SignupLiteResponse(BaseModel):
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
    email: EmailStr
    phone: Optional[str]
    assessments: List[AssessmentSummary]


# =========================================
# دوال التعامل مع Supabase
# =========================================
def get_or_create_user(name: str, email: str, phone: Optional[str] = None) -> str:
    """
    يرجع user_id من جدول users.
    إذا الإيميل موجود يرجع نفس المستخدم، وإذا غير موجود ينشئ مستخدم جديد.
    """
    resp = supabase.table("users").select("id, name, email, phone") \
        .eq("email", email).limit(1).execute()

    data = resp.data or []
    if data:
        # ممكن نحدّث الاسم/الجوال لو حاب
        user = data[0]
        update_payload = {}
        if name and name != user.get("name"):
            update_payload["name"] = name
        if phone and phone != user.get("phone"):
            update_payload["phone"] = phone

        if update_payload:
            supabase.table("users").update(update_payload).eq("id", user["id"]).execute()

        return str(user["id"])

    # ما فيه مستخدم بهالإيميل → ننشئ واحد
    insert_resp = supabase.table("users").insert(
        {
            "name": name,
            "email": email,
            "phone": phone,
        }
    ).execute()

    if not insert_resp.data:
        raise RuntimeError("Failed to create user in Supabase.")

    return str(insert_resp.data[0]["id"])


def save_assessment(
    user_id: Optional[str],
    track: str,
    result: Dict,
) -> None:
    """
    يحفظ نتيجة التقييم في جدول assessments لو فيه user_id.
    """
    if not user_id:
        # لو ما عندنا user_id (زائر بدون تسجيل)، نتجاوز التخزين بهدوء
        return

    supabase.table("assessments").insert(
        {
            "user_id": user_id,
            "track": track,
            "overall_score": result["overallScore"],
            "skill_scores": result["skillScores"],
            "bench_scores": result["benchScores"],
            "ai_report": result["aiReport"],
        }
    ).execute()


def get_user_profile(user_id: str) -> UserProfile:
    # نجيب بيانات المستخدم
    user_resp = supabase.table("users").select(
        "id, name, email, phone"
    ).eq("id", user_id).limit(1).execute()

    if not user_resp.data:
        raise HTTPException(status_code=404, detail="User not found")

    user = user_resp.data[0]

    # نجيب آخر الاختبارات
    assessments_resp = (
        supabase.table("assessments")
        .select("id, created_at, track, overall_score, skill_scores")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    assessments_list: List[AssessmentSummary] = []
    for row in assessments_resp.data or []:
        assessments_list.append(
            AssessmentSummary(
                id=row["id"],
                created_at=datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                ),
                track=row["track"],
                overallScore=row["overall_score"],
                skillScores=row["skill_scores"],
            )
        )

    return UserProfile(
        id=str(user["id"]),
        name=user["name"],
        email=user["email"],
        phone=user.get("phone"),
        assessments=assessments_list,
    )


# =========================================
# منطق التحليل (GPT)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لاحقاً خله على دومين الفرونت فقط
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


@app.post("/signup-lite", response_model=SignupLiteResponse)
def signup_lite(payload: SignupLitePayload):
    """
    يستقبل الاسم + الإيميل + الجوال من الـ Pop-up
    ويرجع userId يصلح نستخدمه في الفرونت وفي /analyze و /profile.
    """
    try:
        user_id = get_or_create_user(
            name=payload.name,
            email=payload.email,
            phone=payload.phone,
        )
        return SignupLiteResponse(userId=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(payload: AnalyzePayload):
    # نحدد user_id لو متوفر
    user_id = payload.userId

    # لو ما فيه userId لكن فيه إيميل، ننشئ/نجيب المستخدم
    if not user_id and payload.email:
        try:
            user_id = get_or_create_user(
                name=payload.name,
                email=payload.email,
                phone=payload.phone,
            )
        except Exception as e:
            # نكمل التحليل حتى لو تخزين المستخدم فشل
            print("Supabase user error:", e)

    answers_list: List[Dict] = [a.dict() for a in payload.answers]
    result = analyze_user(
        name=payload.name,
        track=payload.track,
        answers=answers_list,
    )

    # نخزن النتيجة في جدول assessments لو عندنا user_id
    try:
        save_assessment(user_id=user_id, track=payload.track, result=result)
    except Exception as e:
        print("Supabase assessment error:", e)

    return result


@app.get("/profile/{user_id}", response_model=UserProfile)
def profile_endpoint(user_id: str):
    """
    يرجع بروفايل المستخدم + آخر الاختبارات (للصفحة الخاصة فيه في الفرونت).
    """
    try:
        profile = get_user_profile(user_id)
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# للتشغيل المحلي مباشرة: python main.py
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
