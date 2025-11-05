# main.py
from typing import List, Dict, Optional
import os
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# =========================================
# تحميل المتغيرات من .env (محلياً) + بيئة Render
# =========================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")  # example: https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# بس نطبع تحذير لو في شي ناقص، ما نطيح السيرفر
missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not SUPABASE_URL:
    missing_vars.append("SUPABASE_URL")
if not SUPABASE_SERVICE_ROLE_KEY:
    missing_vars.append("SUPABASE_SERVICE_ROLE_KEY")

if missing_vars:
    print("WARNING: Missing environment variables:", ", ".join(missing_vars))

# REST base URL (لو ما فيه SUPABASE_URL نخليه فاضي ونتعامل معه داخل الفنكشنز)
SUPABASE_REST_URL = (
    SUPABASE_URL.rstrip("/") + "/rest/v1" if SUPABASE_URL else ""
)


def supabase_headers() -> Dict[str, str]:
    """
    يرجع الهيدرز الجاهزة لـ Supabase REST.
    لو المتغيرات ناقصة يرجع Error واضح بدل ما يطيح السيرفر.
    """
    if not SUPABASE_SERVICE_ROLE_KEY or not SUPABASE_REST_URL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: Supabase env vars are missing.",
        )

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# =========================================
# بيانات الـ Benchmark
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
# Helpers مع Supabase REST
# =========================================
def get_user_by_email(email: str) -> Optional[Dict]:
    if not SUPABASE_REST_URL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: SUPABASE_URL is missing.",
        )

    url = f"{SUPABASE_REST_URL}/users"
    params = {
        "select": "id,name,email,phone",
        "email": f"eq.{email}",
    }
    resp = requests.get(url, headers=supabase_headers(), params=params)
    if not resp.ok:
        print("Supabase get_user_by_email ERROR:", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Supabase error (get user)")

    data = resp.json()
    if not data:
        return None
    return data[0]


def create_user(name: str, email: str, phone: Optional[str]) -> str:
    if not SUPABASE_REST_URL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: SUPABASE_URL is missing.",
        )

    url = f"{SUPABASE_REST_URL}/users"
    payload = [{"name": name, "email": email, "phone": phone}]
    resp = requests.post(url, headers=supabase_headers(), json=payload)
    if not resp.ok:
        print("Supabase create_user ERROR:", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Supabase error (create user)")

    data = resp.json()
    return data[0]["id"]


def update_user(user_id: str, name: str, phone: Optional[str]) -> None:
    if not SUPABASE_REST_URL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: SUPABASE_URL is missing.",
        )

    url = f"{SUPABASE_REST_URL}/users"
    params = {"id": f"eq.{user_id}"}
    payload = {"name": name, "phone": phone}
    resp = requests.patch(url, headers=supabase_headers(), params=params, json=payload)
    if not resp.ok:
        print("Supabase update_user ERROR:", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Supabase error (update user)")


def insert_assessment(
    user_id: str,
    track: str,
    overall_score: float,
    skill_scores: Dict[str, float],
    bench_scores: Dict[str, int],
    ai_report: str,
) -> None:
    if not SUPABASE_REST_URL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: SUPABASE_URL is missing.",
        )

    url = f"{SUPABASE_REST_URL}/assessments"
    payload = [
        {
            "user_id": user_id,
            "track": track,
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "bench_scores": bench_scores,
            "ai_report": ai_report,
        }
    ]
    resp = requests.post(url, headers=supabase_headers(), json=payload)
    if not resp.ok:
        print("Supabase insert_assessment ERROR:", resp.status_code, resp.text)
        raise HTTPException(
            status_code=500, detail="Supabase error (insert assessment)"
        )


def fetch_profile_from_supabase(user_id: str) -> UserProfile:
    if not SUPABASE_REST_URL:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: SUPABASE_URL is missing.",
        )

    # User
    url_user = f"{SUPABASE_REST_URL}/users"
    params_user = {
        "select": "id,name,email,phone",
        "id": f"eq.{user_id}",
    }
    resp_user = requests.get(url_user, headers=supabase_headers(), params=params_user)
    if not resp_user.ok:
        print(
            "Supabase fetch_profile user ERROR:",
            resp_user.status_code,
            resp_user.text,
        )
        raise HTTPException(status_code=500, detail="Supabase error (get user)")

    user_data = resp_user.json()
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    user = user_data[0]

    # Assessments
    url_assess = f"{SUPABASE_REST_URL}/assessments"
    params_assess = {
        "select": "id,track,overall_score,skill_scores,created_at",
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc",
    }
    resp_assess = requests.get(
        url_assess, headers=supabase_headers(), params=params_assess
    )
    if not resp_assess.ok:
        print(
            "Supabase fetch_profile assessments ERROR:",
            resp_assess.status_code,
            resp_assess.text,
        )
        raise HTTPException(
            status_code=500, detail="Supabase error (get assessments)"
        )

    rows = resp_assess.json() or []

    assessments: List[AssessmentSummary] = []
    for row in rows:
        created_raw = row["created_at"]
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


# =========================================
# OpenAI helper: تهيئة العميل وقت الحاجة فقط
# =========================================
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: OPENAI_API_KEY is missing.",
        )
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except TypeError as e:
        # هنا لو رجع نفس خطأ proxies راح يبان في اللوق لكن ما يطيح السيرفر أثناء الـ import
        print("OpenAI client init error:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="OpenAI client init error. Please contact support.",
        )


# =========================================
# منطق التحليل + استدعاء GPT
# =========================================
def analyze_user(name: str, track: str, answers: List[Dict]) -> Dict:
    """
    answers: list of {"text": "..", "skill": "Technical", "score": 1..5}
    """
    client = get_openai_client()

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

    overall = (
        round(sum(skill_scores.values()) / len(skill_scores), 1)
        if skill_scores
        else 0.0
    )
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
    version="1.2.0",
    description="Backend for Growday AI Skill Assessment + user profiles",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # بعدين حصرها على دومين الفرونت
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
    try:
        existing = get_user_by_email(payload.email)
        if existing:
            update_user(existing["id"], payload.name, payload.phone)
            return {"userId": existing["id"]}

        user_id = create_user(payload.name, payload.email, payload.phone)
        return {"userId": user_id}

    except HTTPException:
        raise
    except Exception as e:
        # عشان لو صار خطأ ثاني يطلع في اللوق
        print("signup_lite UNEXPECTED ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Unexpected error in signup-lite")


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
    try:
        if payload.userId:
            insert_assessment(
                user_id=payload.userId,
                track=payload.track,
                overall_score=result["overallScore"],
                skill_scores=result["skillScores"],
                bench_scores=result["benchScores"],
                ai_report=result["aiReport"],
            )
    except HTTPException:
        raise
    except Exception as e:
        print("insert_assessment UNEXPECTED ERROR:", repr(e))
        # ما نوقف اليوزر عن الحصول على النتيجة، بس نسجّل اللّوغ

    return result


# 3) /profile/{user_id}  -> يرجّع بروفايل العميل + قائمة اختبارات
@app.get("/profile/{user_id}", response_model=UserProfile)
async def get_profile(user_id: str):
    try:
        profile = fetch_profile_from_supabase(user_id)
        return profile
    except HTTPException:
        raise
    except Exception as e:
        print("get_profile UNEXPECTED ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Unexpected error in profile")


# للتشغيل المحلي: python main.py
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
