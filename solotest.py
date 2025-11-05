import streamlit as st
import openai
from dotenv import load_dotenv
import os
import plotly.graph_objects as go

# --- Load API key ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Page Config ---
st.set_page_config(
    page_title="Growday Skill Test",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Header ---
st.title("🧭 Growday Skill Assessment")
st.markdown("""
**اكتشف مهاراتك الحقيقية في دقائق.**
استنادًا إلى أحدث معايير تحليل الكفاءات العالمية، يقدم لك Growday رؤية عميقة حول نقاط قوتك، ومجالات تطويرك، وخطتك للنمو المهني.
""")

# --- Step 1: Input Form ---
st.header("🔹 الأسئلة الأساسية")
name = st.text_input("اسمك الكامل")
job = st.text_input("ما هو مجالك المهني الحالي؟")
goal = st.text_input("ما هو هدفك المهني خلال السنة القادمة؟")

st.markdown("### 💡 قيّم نفسك من 1 إلى 5:")
q1 = st.slider("مهارات التحليل واتخاذ القرار", 1, 5)
q2 = st.slider("مهارات التواصل والإقناع", 1, 5)
q3 = st.slider("إدارة الوقت والإنجاز", 1, 5)
q4 = st.slider("المهارات التقنية في مجالك", 1, 5)
q5 = st.slider("القدرة على التعلم والتطور", 1, 5)

if st.button("ابدأ التحليل 🧠"):
    with st.spinner("⏳ جاري تحليل بياناتك..."):
        prompt = f"""
        قم بتحليل هذا الشخص مهنياً بناءً على إجاباته:
        الاسم: {name}
        الوظيفة: {job}
        الهدف: {goal}
        تقييماته:
        - التحليل واتخاذ القرار: {q1}/5
        - التواصل والإقناع: {q2}/5
        - إدارة الوقت: {q3}/5
        - المهارات التقنية: {q4}/5
        - التعلم والتطور: {q5}/5

        أجب بالنتائج التالية:
        1. تحليل عام عن شخصيته المهنية (200 كلمة)
        2. 3 نقاط قوة
        3. 3 نقاط تحتاج لتطوير
        4. خطة تطوير ذكية لمدة 30 يوم
        5. مقياس تناسبه مع مهن أخرى (0–100)
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional career coach."},
                {"role": "user", "content": prompt},
            ]
        )

        result = response.choices[0].message.content

        # --- Display Results ---
        st.subheader("📊 نتائجك الشخصية")
        st.write(result)

        # --- Visualization ---
        skills = ["تحليل القرار", "التواصل", "الوقت", "التقنية", "التعلم"]
        scores = [q1, q2, q3, q4, q5]

        fig = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=skills,
            fill='toself',
            name='تقييمك'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            title="📈 خريطة مهاراتك المهنية"
        )
        st.plotly_chart(fig)

        st.success("✨ تم تحليل بياناتك بنجاح!")
