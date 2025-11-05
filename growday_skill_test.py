import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF

# ========== CONFIG ==========
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Arabic font path for PDF (put the TTF file inside ./fonts)
FONT_PATH = os.path.join(
    os.path.dirname(__file__),
    "fonts",
    "NotoNaskhArabic-Regular.ttf",  # change name if you use another font
)

# ========== GLOBAL CSS (Growday style) ==========
GROWDAY_CSS = """
<style>
body {
    background: radial-gradient(circle at top left, #e0f2fe 0, #f9fafb 45%, #f5f3ff 100%);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.main-card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 24px;
    padding: 24px 28px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.1;
}

.hero-sub {
    font-size: 16px;
    color: #4b5563;
}

.grow-btn-primary {
    background: linear-gradient(90deg, #2563eb, #06b6d4, #8b5cf6);
    color: white;
    padding: 12px 26px;
    border-radius: 999px;
    border: none;
    font-weight: 600;
    cursor: pointer;
}

.grow-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.08);
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 600;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(15, 118, 110, 0.06);
    color: #0f766e;
    font-size: 13px;
    font-weight: 500;
    margin-right: 8px;
}

.section-title {
    font-weight: 700;
    font-size: 22px;
    margin-bottom: 4px;
}

.section-sub {
    font-size: 14px;
    color: #6b7280;
}

</style>
"""

# ========== QUESTION BANK ==========
# Each track has list of dicts: text, skill
TRACK_QUESTIONS: Dict[str, List[Dict[str, str]]] = {
    "Data Analyst": [
        {"skill": "Technical", "text": "How confident are you in using SQL to join and aggregate data from multiple tables?"},
        {"skill": "Technical", "text": "How comfortable are you building dashboards in tools like Power BI, Tableau, or Looker?"},
        {"skill": "Technical", "text": "How well can you clean messy data (missing values, outliers, inconsistent formats) before analysis?"},
        {"skill": "Business", "text": "How well do you translate a business question into a clear analysis plan with metrics and hypotheses?"},
        {"skill": "Business", "text": "How comfortable are you working with non-technical stakeholders (product, marketing, finance)?"},
        {"skill": "Communication", "text": "How strong are you at presenting insights to executives in a clear and concise way?"},
        {"skill": "Communication", "text": "How good are you at telling a story with data (context, insight, recommendation)?"},
        {"skill": "Problem Solving", "text": "When dashboards break or numbers look wrong, how fast can you debug and find the root cause?"},
        {"skill": "Problem Solving", "text": "How comfortable are you designing A/B tests or experiments and interpreting the results?"},
        {"skill": "Problem Solving", "text": "How well do you prioritize analysis requests based on impact vs. effort?"},
    ],
    "Product Manager": [
        {"skill": "Business", "text": "How well do you define product goals and success metrics aligned with company strategy?"},
        {"skill": "Business", "text": "How confident are you in building business cases (revenue, cost, ROI) for new features?"},
        {"skill": "Customer", "text": "How strong is your ability to run user interviews and extract real pain points?"},
        {"skill": "Customer", "text": "How consistently do you use data and UX research together to drive decisions?"},
        {"skill": "Execution", "text": "How good are you at writing clear PRDs / user stories that developers can execute on?"},
        {"skill": "Execution", "text": "How well do you manage trade-offs between scope, quality, and deadlines?"},
        {"skill": "Stakeholder", "text": "How comfortable are you aligning engineering, design, and business stakeholders around one roadmap?"},
        {"skill": "Stakeholder", "text": "How strong are you at handling conflicts and pushback from senior stakeholders?"},
        {"skill": "Leadership", "text": "How often do team members look to you for direction even if you have no formal authority?"},
        {"skill": "Leadership", "text": "How good are you at communicating a compelling product vision?"},
    ],
    "Software Engineer": [
        {"skill": "Technical", "text": "How confident are you in writing clean, well-structured, and testable code?"},
        {"skill": "Technical", "text": "How strong is your understanding of algorithms, data structures, and complexity?"},
        {"skill": "Technical", "text": "How comfortable are you working with cloud platforms (AWS, Azure, GCP)?"},
        {"skill": "Quality", "text": "How consistently do you write unit/integration tests and use CI/CD pipelines?"},
        {"skill": "Quality", "text": "How good are you at debugging complex production issues?"},
        {"skill": "Collaboration", "text": "How effective are you in code reviews (giving and receiving feedback)?"},
        {"skill": "Collaboration", "text": "How comfortable are you collaborating with product and design to clarify requirements?"},
        {"skill": "Architecture", "text": "How strong is your ability to design system architecture for scalability and reliability?"},
        {"skill": "Growth", "text": "How actively do you learn new frameworks, tools, or languages relevant to your stack?"},
        {"skill": "Growth", "text": "How often do you contribute to improving team processes or technical direction?"},
    ],
}

# ========== BENCHMARKS (approx Saudi / GCC market) ==========
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


# ========== PDF CREATION (Arabic font) ==========
class GrowdayPDF(FPDF):
    def header(self):
        if "GrowArabic" in self.fonts:
            self.set_font("GrowArabic", "", 16)
        else:
            self.set_font("Helvetica", "", 16)
        self.cell(0, 10, "تقرير Growday لمهاراتك المهنية", ln=1, align="C")
        self.ln(4)


def create_pdf_report(
    name: str,
    track: str,
    report_text: str,
    user_skill_scores: Dict[str, float],
    bench_scores: Dict[str, float],
) -> bytes:
    pdf = GrowdayPDF()
    pdf.add_page()

    # register arabic font
    if os.path.exists(FONT_PATH):
        pdf.add_font("GrowArabic", "", FONT_PATH, uni=True)
        pdf.set_font("GrowArabic", "", 12)
    else:
        pdf.set_font("Helvetica", "", 12)

    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.multi_cell(0, 8, f"الاسم: {name}")
    pdf.multi_cell(0, 8, f"المسار المهني: {track}")
    pdf.ln(4)

    pdf.set_font(pdf.font_family, "", 13)
    pdf.multi_cell(0, 7, "ملخص الذكاء الاصطناعي عن وضعك الحالي:")
    pdf.ln(2)

    pdf.set_font(pdf.font_family, "", 11)
    pdf.multi_cell(0, 6, report_text)
    pdf.ln(4)

    pdf.set_font(pdf.font_family, "", 13)
    pdf.multi_cell(0, 7, "درجاتك مقارنة بمتوسط السوق:")
    pdf.ln(2)

    pdf.set_font(pdf.font_family, "", 11)
    for skill, user_score in user_skill_scores.items():
        bench_score = bench_scores.get(skill, 0)
        line = f"- {skill}: درجتك {int(user_score)} / 100 · السوق {int(bench_score)} / 100"
        pdf.multi_cell(0, 6, line)

    pdf.ln(4)
    pdf.multi_cell(
        0,
        6,
        "نقترح أن تركز خلال 30 يومًا على مهارتين إلى ثلاث مهارات رئيسية لرفع درجتك العامة.",
    )

    pdf_bytes = pdf.output(dest="S").encode("latin1", "ignore")
    return pdf_bytes


# ========== ANALYTICS & SCORING ==========
def compute_skill_scores(track: str, answers: Dict[int, int]) -> Dict[str, float]:
    questions = TRACK_QUESTIONS[track]
    skill_buckets: Dict[str, List[int]] = {}
    for idx, q in enumerate(questions):
        score = answers.get(idx, 0)
        skill = q["skill"]
        skill_buckets.setdefault(skill, []).append(score)

    skill_scores: Dict[str, float] = {}
    for skill, values in skill_buckets.items():
        if values:
            avg = sum(values) / len(values)
            skill_scores[skill] = round(avg / 5 * 100, 1)
        else:
            skill_scores[skill] = 0.0
    return skill_scores


def compute_overall(skill_scores: Dict[str, float]) -> float:
    if not skill_scores:
        return 0.0
    return round(sum(skill_scores.values()) / len(skill_scores), 1)


def estimate_percentile(overall: float, bench_overall: float) -> int:
    # rough estimate just for UX
    diff = overall - bench_overall  # + means above market
    percentile = 50 + diff * 1.2  # 10 points diff → ~12 percentile
    percentile = max(5, min(95, int(percentile)))
    return percentile


# ========== OPENAI REPORT ==========
def generate_ai_report(
    name: str,
    track: str,
    skill_scores: Dict[str, float],
    bench_scores: Dict[str, float],
) -> str:
    try:
        user_json = {
            "name": name,
            "track": track,
            "skills": skill_scores,
            "market_benchmark": bench_scores,
        }

        system_prompt = (
            "You are a senior career coach in the GCC market. "
            "Write the answer in clear Arabic (Saudi / GCC tone). "
            "The user is a professional who just took a skill test. "
            "You must:\n"
            "1) Summarize their strengths.\n"
            "2) Highlight 2–3 priority gaps vs. the local market.\n"
            "3) Give a focused 30-day learning plan (weekly actions).\n"
            "4) Keep the tone motivating but realistic.\n"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"هذي بيانات الاختبار:\n{user_json}\nاكتب تقرير مهني مختصر ومنظم.",
                },
            ],
            temperature=0.4,
        )

        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            "واجهنا مشكلة بسيطة مع خدمة الذكاء الاصطناعي حاليًا، "
            "لكن بشكل عام درجاتك تعطي صورة جيدة عن نقاط قوتك وضعفك. "
            "ركّز في الفترة الجاية على رفع المهارات الأقل من متوسط السوق، "
            "وابنِ خطة أسبوعية واضحة (تعلم، تطبيق عملي، ثم مراجعة)."
        )


# ========== VISUALIZATION ==========
def render_results(
    name: str,
    track: str,
    user_skill_scores: Dict[str, float],
    bench_scores: Dict[str, float],
    overall_score: float,
):
    st.markdown("### 🎯 Your Growday Skill Snapshot")

    skills = list(user_skill_scores.keys())
    user_vals = [user_skill_scores[s] for s in skills]
    bench_vals = [bench_scores.get(s, 0) for s in skills]

    bench_overall = compute_overall(bench_scores)
    percentile = estimate_percentile(overall_score, bench_overall)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "indicator"}]],
        column_widths=[0.65, 0.35],
    )

    # Radar: you vs market
    fig.add_trace(
        go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=skills + [skills[0]],
            name="You",
            fill="toself",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatterpolar(
            r=bench_vals + [bench_vals[0]],
            theta=skills + [skills[0]],
            name="Market",
            line=dict(width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            title={"text": "Overall Skill Index"},
            delta={"reference": bench_overall},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
            },
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", x=0.2, y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Bar comparison
    st.markdown("### 📊 Skill-by-skill comparison")

    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(x=skills, y=user_vals, name="You", marker_color="#2563eb")
    )
    bar_fig.add_trace(
        go.Bar(x=skills, y=bench_vals, name="Market", marker_color="#9ca3af")
    )
    bar_fig.update_layout(
        barmode="group",
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown(
        f"""
✅ **{name}**، درجتك العامة في مسار **{track}** هي **{overall_score} / 100**  
أنت تقريبًا في **المئين {percentile}** مقارنة بالمهنيين في هذا المسار في السوق المحلي (تقدير تقريبي).

ركّز خلال الأسابيع الجاية على المهارات اللي أقل من متوسط السوق،  
خصوصًا أي مهارة أقل من **60/100**.
        """
    )


# ========== UI ==========
def main():
    st.set_page_config(
        page_title="Growday Skill Test",
        page_icon="🧠",
        layout="wide",
    )
    st.markdown(GROWDAY_CSS, unsafe_allow_html=True)

    # HERO
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown('<div class="grow-badge">Growday · AI Skill Coach</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Discover your real skills.<br>Unlock your next level.</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">'
            "Growday uses AI to analyze your strengths, compare you with the local market, "
            "and generate a personalized 30-day growth plan."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="metric-pill">✅ Built for professionals in the Middle East</div>
            <div class="metric-pill">📊 Based on global skill frameworks</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Your Skill Test</div>
            <div class="section-sub">
            Choose your track, answer a short set of questions, and we'll compare you to the Saudi / GCC market.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 🧠 Growday Skill Test")

    with st.form("skill_test_form"):
        st.markdown('<div class="main-card">', unsafe_allow_html=True)

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            name = st.text_input("Full name", "")
            email = st.text_input("Email (for sending your PDF report)", "")
        with info_col2:
            experience = st.selectbox(
                "Years of experience",
                ["0–1", "2–4", "5–7", "8–12", "13+"],
            )
            track = st.selectbox(
                "Choose your track",
                list(TRACK_QUESTIONS.keys()),
            )

        st.markdown("### Answer the questions (1 = low, 5 = very strong)")

        answers: Dict[int, int] = {}
        questions = TRACK_QUESTIONS[track]
        for idx, q in enumerate(questions):
            label = f"{q['text']}  \n*(Skill: {q['skill']})*"
            answers[idx] = st.slider(label, 1, 5, 3)

        submitted = st.form_submit_button("Generate my skill report")

        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not name:
            st.error("Please enter your name before generating the report.")
            return

        user_skill_scores = compute_skill_scores(track, answers)
        bench_scores = BENCHMARKS.get(track, {})
        overall_score = compute_overall(user_skill_scores)

        # AI report
        with st.spinner("Analyzing your answers with AI..."):
            report_text = generate_ai_report(name, track, user_skill_scores, bench_scores)

        # Show visuals
        render_results(name, track, user_skill_scores, bench_scores, overall_score)

        # Show AI report
        st.markdown("### 🧾 AI Skill Coach Report (Arabic)")
        st.write(report_text)

        # PDF download
        try:
            pdf_bytes = create_pdf_report(
                name=name,
                track=track,
                report_text=report_text,
                user_skill_scores=user_skill_scores,
                bench_scores=bench_scores,
            )

            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="growday_skill_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(
                "The skill report is ready, but we couldn't generate the PDF file right now. "
                "You can copy the text above or try again later."
            )


if __name__ == "__main__":
    main()
