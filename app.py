
import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
from url_intelligence import analyze_text_urls, check_virustotal_url, calculate_unified_risk
# ==========================================
# 1. إعداد الصفحة
# ==========================================

st.set_page_config(
    page_title="ArabShield AI Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. مسارات النموذج
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "arabshield_lstm_v2.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "arabshield_lstm_tokenizer_v2.pkl"

MAX_LEN = 150


# مسار قاعدة البيانات
DATABASE_PATH = BASE_DIR / "arabshield.db"


# ==========================================
# قاعدة بيانات سجل التحليلات
# ==========================================

def init_database():
    """إنشاء جدول سجل التحليلات عند تشغيل التطبيق."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation TEXT NOT NULL,
                prediction TEXT NOT NULL,
                scam_probability REAL NOT NULL,
                safe_probability REAL NOT NULL,
                analysis_date TEXT NOT NULL
            )
        """)
        connection.commit()


def save_analysis(conversation, predicted_label, scam_probability):
    """حفظ نتيجة التحليل في قاعدة البيانات."""
    prediction = "احتيال محتمل" if predicted_label == 1 else "محادثة سليمة"
    safe_probability = 1.0 - scam_probability
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO analysis_history
            (conversation, prediction, scam_probability, safe_probability, analysis_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conversation,
            prediction,
            float(scam_probability),
            float(safe_probability),
            analysis_date
        ))
        connection.commit()


def get_analysis_history(search_text=""):
    """استرجاع سجل التحليلات، مع إمكانية البحث داخل المحادثات والنتائج."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        if search_text.strip():
            search_value = f"%{search_text.strip()}%"
            rows = connection.execute("""
                SELECT * FROM analysis_history
                WHERE conversation LIKE ? OR prediction LIKE ?
                ORDER BY id DESC
            """, (search_value, search_value)).fetchall()
        else:
            rows = connection.execute("""
                SELECT * FROM analysis_history
                ORDER BY id DESC
            """).fetchall()

    return [dict(row) for row in rows]


def delete_analysis(record_id):
    """حذف سجل تحليل محدد."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "DELETE FROM analysis_history WHERE id = ?",
            (int(record_id),)
        )
        connection.commit()


def clear_analysis_history():
    """حذف جميع سجلات التحليلات."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM analysis_history")
        connection.commit()


# إنشاء قاعدة البيانات والجدول تلقائيًا
init_database()


# ==========================================
# 3. إعداد LIME
# ==========================================

def arabic_splitter(text):
    return re.findall(r'\S+', text)


explainer = LimeTextExplainer(
    class_names=["محادثة سليمة", "احتيال"],
    split_expression=arabic_splitter,
    random_state=42
)


# ==========================================
# 4. تحميل النموذج والـ Tokenizer
# ==========================================

@st.cache_resource
def load_model_and_tokenizer():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ملف النموذج غير موجود: {MODEL_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"ملف Tokenizer غير موجود: {TOKENIZER_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


# ==========================================
# 5. معالجة النص العربي
# ==========================================

def normalize_arabic_text_v2(text):

    text = str(text)

    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    text = text.replace("ـ", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 6. تحليل محادثة واحدة باستخدام LSTM
# ==========================================

def predict_conversation(text, model, tokenizer):

    normalized_text = normalize_arabic_text_v2(text)

    sequence = tokenizer.texts_to_sequences(
        [normalized_text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(
            padded_sequence,
            verbose=0
        )[0][0]
    )

    return probability


# ==========================================
# 7. التنبؤ بعدة نصوص لاستخدام LIME
# ==========================================

def predict_with_lstm(texts, model, tokenizer):

    normalized_texts = [
        normalize_arabic_text_v2(text)
        for text in texts
    ]

    sequences = tokenizer.texts_to_sequences(
        normalized_texts
    )

    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    scam_probabilities = model.predict(
        padded_sequences,
        verbose=0
    ).flatten()

    return np.column_stack([
        1 - scam_probabilities,
        scam_probabilities
    ])


# ==========================================
# 8. تفسير المحادثة باستخدام LIME
# ==========================================

def explain_conversation(text, model, tokenizer):

    probabilities = predict_with_lstm(
        [text],
        model,
        tokenizer
    )[0]

    predicted_label = int(
        probabilities[1] >= 0.5
    )

    explanation = explainer.explain_instance(
        text_instance=text,

        classifier_fn=lambda texts: predict_with_lstm(
            texts,
            model,
            tokenizer
        ),

        labels=(predicted_label,),
        num_features=10,
        num_samples=1000
    )

    return explanation.as_list(
        label=predicted_label
    )


# ==========================================
# 9. تحويل نتائج LIME إلى جدول
# ==========================================

def prepare_explanation_data(important_words, predicted_label):

    rows = []

    for word, weight in important_words:

        # توحيد اتجاه التأثير:
        # موجب = يدعم الاحتيال
        # سالب = يعارض الاحتيال

        if predicted_label == 1:
            scam_effect = float(weight)
        else:
            scam_effect = -float(weight)

        if scam_effect > 0:
            interpretation = "يدعم الاحتيال"

        elif scam_effect < 0:
            interpretation = "يعارض الاحتيال"

        else:
            interpretation = "تأثير محايد"

        rows.append({
            "الكلمة": str(word),
            "قوة التأثير": round(abs(float(weight)), 4),
            "الاتجاه": interpretation,
            "تأثير الاحتيال": round(scam_effect, 4)
        })

    return pd.DataFrame(
        rows,
        columns=[
            "الكلمة",
            "قوة التأثير",
            "الاتجاه",
            "تأثير الاحتيال"
        ]
    )


# ==========================================
# 10. عرض نتائج LIME
# ==========================================

def display_lime_results(important_words, predicted_label):

    if not important_words:

        st.info("لم يتم العثور على كلمات مؤثرة.")

        return

    word_df = prepare_explanation_data(
        important_words,
        predicted_label
    )

    if word_df.empty:

        st.info("لا توجد بيانات لعرضها.")

        return

    # جدول الكلمات

    st.markdown("### 📋 جدول الكلمات المؤثرة")

    display_df = word_df[
        ["الكلمة", "قوة التأثير", "الاتجاه"]
    ].copy()

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

    # الرسم البياني

    st.markdown("### 📊 الرسم البياني لتأثير الكلمات")

    chart_df = word_df[
        ["الكلمة", "تأثير الاحتيال"]
    ].copy()

    chart_df = chart_df.set_index("الكلمة")

    st.bar_chart(
        chart_df,
        width="stretch"
    )

    st.caption(
        "القيم الموجبة تدعم تصنيف الاحتيال، "
        "والقيم السالبة تعارض تصنيف الاحتيال."
    )

    # ملخص التفسير

    scam_words = word_df[
        word_df["تأثير الاحتيال"] > 0
    ]["الكلمة"].tolist()

    safe_words = word_df[
        word_df["تأثير الاحتيال"] < 0
    ]["الكلمة"].tolist()

    st.markdown("### 📝 ملخص تفسير النموذج")

    if scam_words:

        st.error(
            "الكلمات التي دعمت الاشتباه بالاحتيال: "
            + "، ".join(scam_words)
        )

    if safe_words:

        st.success(
            "الكلمات التي عارضت الاشتباه بالاحتيال: "
            + "، ".join(safe_words)
        )


# ==========================================
# 11. تنسيق الواجهة
# ==========================================

st.markdown("""
<style>

    .stApp {
        direction: rtl;
        text-align: right;
    }

    h1, h2, h3, p, label {
        text-align: right;
    }

    .main-title {
        text-align: center;
        color: #19B5A5;
        font-size: 38px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 17px;
    }

    .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# 12. عنوان النظام
# ==========================================

st.markdown(
    '<div class="main-title">🛡️ ArabShield AI Pro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'نظام ذكي لاكتشاف الاحتيال الإلكتروني باللغة العربية'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 13. القائمة الجانبية
# ==========================================

with st.sidebar:

    st.title("🛡️ ArabShield")

    st.caption("AI Cybersecurity System")

    st.divider()

    page = st.radio(
        "اختر الصفحة",
        [
            "🏠 الرئيسية",
            "🔍 تحليل المحادثات",
            "📚 سجل التحليلات",
            "ℹ️ حول النظام"
        ]
    )

    st.divider()

    st.caption("ArabShield AI Pro © 2026")


# ==========================================
# 14. الصفحة الرئيسية
# ==========================================

if page == "🏠 الرئيسية":

    st.header("مرحباً بك في ArabShield AI Pro")

    st.write("""
    يساعدك النظام على تحليل المحادثات العربية واكتشاف
    المؤشرات المحتملة للاحتيال باستخدام الذكاء الاصطناعي.
    """)

    st.subheader("📝 أدخل المحادثة المراد تحليلها")

    conversation = st.text_area(
        "نص المحادثة",
        placeholder="الصق المحادثة العربية هنا...",
        height=200
    )

    if st.button(
        "🔍 تحليل المحادثة",
        width="stretch"
    ):

        if not conversation.strip():

            st.warning(
                "يرجى إدخال نص المحادثة أولاً."
            )

        else:

            try:

                # تحميل النموذج

                with st.spinner("جاري تحميل النموذج..."):

                    model, tokenizer = load_model_and_tokenizer()

                # تحليل المحادثة

                with st.spinner("جاري تحليل المحادثة..."):

                    scam_probability = predict_conversation(
                        conversation,
                        model,
                        tokenizer
                    )

                predicted_label = int(
                    scam_probability >= 0.5
                )

                # حفظ نتيجة التحليل تلقائيًا في سجل التحليلات
                save_analysis(
                    conversation,
                    predicted_label,
                    scam_probability
                )

                # ==================================
                # نتيجة التحليل
                # ==================================

                st.divider()

                st.subheader("📊 نتيجة التحليل")

                if predicted_label == 1:

                    st.error(
                        "⚠️ توجد مؤشرات محتملة على الاحتيال"
                    )

                else:

                    st.success(
                        "✅ لم يكتشف النموذج مؤشرات كافية على الاحتيال"
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "احتمالية الاحتيال",
                        f"{scam_probability * 100:.2f}%"
                    )

                with col2:

                    st.metric(
                        "احتمالية المحادثة السليمة",
                        f"{(1 - scam_probability) * 100:.2f}%"
                    )

                st.progress(scam_probability)

                # ==================================
                # فحص الروابط - ArabShield URL Intelligence
                # ==================================

                st.divider()
                st.subheader("🔗 تحليل الروابط - ArabShield URL Intelligence")

                url_results = analyze_text_urls(conversation)
                vt_results = []

                st.divider()
                st.subheader("🔗 تحليل الروابط - ArabShield URL Intelligence")

                if not url_results:
                    st.info("لم يتم العثور على روابط داخل الرسالة.")
                else:
                    st.caption(
                        "يحلل النظام خصائص الرابط محليًا، ويستعلم عن VirusTotal عند إعداد المفتاح. "
                        "قد تُشارك الروابط مع خدمة خارجية عند تفعيلها."
                    )

                    vt_api_key = st.secrets.get("VIRUSTOTAL_API_KEY", "")
                    for url_index, url_result in enumerate(url_results, start=1):
                        with st.expander(
                            f"الرابط {url_index}: {url_result['host'] or 'تعذر تحديد النطاق'} "
                            f"| مؤشرات محلية: {url_result['level']}",
                            expanded=True
                        ):
                            st.code(url_result["url"], language=None)
                            metric_col1, metric_col2 = st.columns(2)
                            metric_col1.metric("مؤشر الرابط المحلي", f"{url_result['score']}/100")
                            metric_col2.metric("التقييم المحلي", url_result["level"])
                            for flag in url_result["flags"]:
                                st.write(f"• {flag}")

                            if vt_api_key:
                                with st.spinner("جاري التحقق من سمعة الرابط عبر VirusTotal..."):
                                    vt_result = check_virustotal_url(url_result["url"], vt_api_key)
                                vt_results.append(vt_result)
                                st.markdown("**🌐 نتيجة VirusTotal**")
                                if vt_result.get("status") == "report":
                                    stats = vt_result.get("stats", {})
                                    malicious = int(stats.get("malicious", 0))
                                    suspicious = int(stats.get("suspicious", 0))
                                    harmless = int(stats.get("harmless", 0))
                                    undetected = int(stats.get("undetected", 0))
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("ضار", malicious)
                                    c2.metric("مشبوه", suspicious)
                                    c3.metric("سليم", harmless)
                                    c4.metric("لم يُكتشف", undetected)
                                    st.markdown(f"[فتح تقرير VirusTotal]({vt_result['permalink']})")
                                elif vt_result.get("status") == "submitted":
                                    st.info(vt_result["message"])
                                    st.markdown(f"[فتح صفحة الرابط في VirusTotal]({vt_result['permalink']})")
                                else:
                                    st.warning(vt_result.get("message", "تعذر الحصول على نتيجة VirusTotal."))
                            else:
                                vt_results.append({"status": "not_configured"})
                                st.caption("VirusTotal غير مفعّل: أضف VIRUSTOTAL_API_KEY إلى Streamlit Secrets.")

                # ==================================
                # التقييم الموحد للمخاطر
                # ==================================
                unified = calculate_unified_risk(
                    scam_probability,
                    url_results,
                    vt_results
                )

                st.divider()
                st.subheader("🛡️ التقرير الأمني - ArabShield Security Report")

                # Executive summary for non-technical users
                if unified["level"] == "مرتفع":
                    st.error(
                        f"🔴 **الخلاصة: خطورة مرتفعة** — رُصدت مؤشرات تستدعي التعامل مع الرسالة "
                        f"على أنها تهديد محتمل. مؤشر المخاطر: **{unified['score']}/100**."
                    )
                    summary_text = (
                        f"الرسالة تحمل مؤشرات خطورة مرتفعة (مؤشر {unified['score']}/100). "
                        "تجنب فتح الروابط أو مشاركة أي معلومات أو رموز تحقق."
                    )
                elif unified["level"] == "متوسط":
                    st.warning(
                        f"🟠 **الخلاصة: خطورة متوسطة** — توجد مؤشرات تحتاج إلى تحقق إضافي. "
                        f"مؤشر المخاطر: **{unified['score']}/100**."
                    )
                    summary_text = (
                        f"الرسالة تحمل مؤشرات خطورة متوسطة (مؤشر {unified['score']}/100). "
                        "تحقق من الجهة عبر قنواتها الرسمية قبل التفاعل."
                    )
                else:
                    st.success(
                        f"🟢 **الخلاصة: مؤشرات منخفضة** — لم تظهر مؤشرات قوية في الفحوص المتاحة. "
                        f"مؤشر المخاطر: **{unified['score']}/100**."
                    )
                    summary_text = (
                        f"لم تظهر مؤشرات قوية في الفحوص المتاحة (مؤشر {unified['score']}/100). "
                        "هذا لا يضمن سلامة الرسالة أو الرابط."
                    )

                score_col, level_col = st.columns([2, 1])
                score_col.metric("مؤشر المخاطر الموحد", f"{unified['score']}/100")
                level_col.metric("مستوى الخطورة", unified["level"])
                st.progress(unified["score"] / 100)

                st.markdown("### 📊 نتائج محركات التحليل")
                risk_cols = st.columns(3)
                risk_cols[0].metric("🧠 LSTM", f"{unified['lstm_score']}/100")
                risk_cols[1].metric(
                    "🔗 تحليل الرابط",
                    f"{unified['local_score']}/100" if unified["local_score"] is not None else "لا يوجد رابط"
                )
                risk_cols[2].metric(
                    "🌐 VirusTotal",
                    f"{unified['virustotal_score']}/100" if unified["virustotal_score"] is not None else "لا توجد نتيجة"
                )

                st.markdown("### 🔎 لماذا ظهرت هذه النتيجة؟")
                for reason in unified["reasons"]:
                    st.markdown(f"- {reason}")

                # Actionable recommendations based on available findings
                st.markdown("### 🧭 التوصيات الأمنية")
                recommendations = []
                if unified["level"] == "مرتفع":
                    recommendations.extend([
                        "لا تفتح الرابط ولا تحمّل أي ملف منه.",
                        "لا تُدخل كلمة مرور أو رمز تحقق أو بيانات بطاقة/حساب.",
                        "تحقق من العرض أو الرسالة عبر التطبيق أو الموقع الرسمي للجهة، بكتابة عنوانه بنفسك.",
                    ])
                elif unified["level"] == "متوسط":
                    recommendations.extend([
                        "لا تتفاعل مع الرابط حتى تتحقق من مصدر الرسالة.",
                        "تواصل مع الجهة عبر رقمها أو موقعها الرسمي المعروف، وليس عبر بيانات الرسالة.",
                        "لا تشارك رموز التحقق أو بيانات الدفع.",
                    ])
                else:
                    recommendations.extend([
                        "لم تظهر مؤشرات قوية، لكن لا تعتبر ذلك ضمانًا للسلامة.",
                        "تأكد من اسم النطاق والجهة قبل إدخال أي بيانات.",
                        "لا تشارك كلمات المرور أو رموز التحقق عبر روابط تصل في الرسائل.",
                    ])
                if url_results:
                    recommendations.append(
                        "إذا كان الرابط يخص جهة معروفة، افتح موقعها الرسمي بكتابة عنوانه يدويًا بدل استخدام رابط الرسالة."
                    )
                for item in recommendations:
                    st.markdown(f"- {item}")

                st.caption(
                    "هذا المؤشر التجريبي أداة فرز إرشادية وليس احتمالًا إحصائيًا أو حكمًا قاطعًا. "
                    "النتيجة تعتمد على الفحوص المتاحة وقد تتغير مع تحديث تقارير السمعة."
                )

                st.warning(
                    "تنبيه: لا تفتح الروابط المشبوهة، ولا تدخل كلمات المرور أو رموز التحقق أو بيانات الدفع."
                )


                # ==================================
                # تفسير LIME
                # ==================================

                st.divider()

                st.subheader(
                    "🧠 تفسير نتيجة الذكاء الاصطناعي"
                )

                st.write(
                    "الكلمات الأكثر تأثيراً في قرار النموذج:"
                )

                try:

                    with st.spinner(
                        "جاري استخراج الكلمات المؤثرة باستخدام LIME..."
                    ):

                        important_words = explain_conversation(
                            conversation,
                            model,
                            tokenizer
                        )

                    display_lime_results(
                        important_words,
                        predicted_label
                    )

                except Exception as lime_error:

                    st.warning(
                        f"تعذر استخراج تفسير الكلمات: {lime_error}"
                    )

                # ==================================
                # التنبيه الأمني
                # ==================================

                st.divider()

                st.info("""
                🛡️ تنبيه أمني:

                نتيجة النموذج تقديرية وليست إثباتاً قاطعاً
                لوجود الاحتيال أو عدمه.

                تحقق من هوية المتصل قبل تحويل الأموال
                أو مشاركة المعلومات الحساسة.
                """)

            except Exception as error:

                st.error(
                    f"حدث خطأ أثناء التحليل: {error}"
                )


# ==========================================
# 15. صفحة تحليل المحادثات
# ==========================================

elif page == "🔍 تحليل المحادثات":

    st.header("🔍 تحليل المحادثات")

    st.write("""
    يمكنك إدخال المحادثة من الصفحة الرئيسية
    لإجراء التحليل وعرض تفسير النموذج.
    """)


# ==========================================
# 16. سجل التحليلات
# ==========================================

elif page == "📚 سجل التحليلات":

    st.header("📚 سجل التحليلات")
    st.write("يمكنك مراجعة التحليلات السابقة والبحث فيها وحذف السجلات.")

    search_text = st.text_input(
        "🔎 ابحث في المحادثات أو نتائج التحليل",
        placeholder="اكتب كلمة للبحث..."
    )

    history = get_analysis_history(search_text)

    if not history:
        st.info("لا توجد سجلات مطابقة حتى الآن.")
    else:
        total_records = len(get_analysis_history())
        scam_count = sum(
            1 for item in get_analysis_history()
            if item["prediction"] == "احتيال محتمل"
        )
        safe_count = total_records - scam_count

        col1, col2, col3 = st.columns(3)
        col1.metric("إجمالي التحليلات", total_records)
        col2.metric("احتيال محتمل", scam_count)
        col3.metric("محادثات سليمة", safe_count)

        st.divider()

        for item in history:
            title = (
                f'#{item["id"]} | {item["prediction"]} | '
                f'{item["analysis_date"]}'
            )

            with st.expander(title):
                st.markdown("**نص المحادثة:**")
                st.text_area(
                    "المحادثة",
                    value=item["conversation"],
                    height=140,
                    disabled=True,
                    key=f'history_conversation_{item["id"]}'
                )

                col1, col2 = st.columns(2)
                col1.metric(
                    "احتمالية الاحتيال",
                    f'{item["scam_probability"] * 100:.2f}%'
                )
                col2.metric(
                    "احتمالية المحادثة السليمة",
                    f'{item["safe_probability"] * 100:.2f}%'
                )

                if st.button(
                    "🗑️ حذف هذا السجل",
                    key=f'delete_record_{item["id"]}'
                ):
                    delete_analysis(item["id"])
                    st.success("تم حذف السجل.")
                    st.rerun()

        st.divider()
        if st.button("🗑️ حذف جميع السجلات", type="secondary"):
            st.session_state["confirm_clear_history"] = True

        if st.session_state.get("confirm_clear_history", False):
            st.warning("هل أنت متأكد من حذف جميع سجلات التحليلات؟")
            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button("نعم، احذف الجميع", type="primary"):
                    clear_analysis_history()
                    st.session_state["confirm_clear_history"] = False
                    st.success("تم حذف جميع السجلات.")
                    st.rerun()

            with cancel_col:
                if st.button("إلغاء"):
                    st.session_state["confirm_clear_history"] = False
                    st.rerun()


# ==========================================
# 16. صفحة حول النظام
# ==========================================

elif page == "ℹ️ حول النظام":

    st.markdown(
        """
        <div style="text-align:center; padding: 12px 8px 20px;">
            <div style="font-size:42px;">🛡️</div>
            <h1 style="margin:4px 0 8px; color:#087F8C;">ArabShield AI Pro</h1>
            <p style="font-size:16px; color:#64748b; margin:0;">
                نظام ذكي لتحليل مؤشرات الاحتيال الإلكتروني في المحادثات العربية
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background:rgba(8,127,140,.07); border:1px solid rgba(8,127,140,.18);
                    border-radius:14px; padding:18px 20px; line-height:1.9; text-align:right;">
            <h3 style="margin-top:0; color:#087F8C;">نبذة عن النظام</h3>
            <p>يهدف <b>ArabShield AI Pro</b> إلى اكتشاف مؤشرات الاحتيال الإلكتروني
            في المحادثات العربية باستخدام تقنيات التعلم العميق وشبكات LSTM.</p>
            <p>يحلل النظام النصوص ويستخدم تقنية <b>LIME</b> لتفسير الكلمات التي
            ساهمت في قرار التصنيف، كما يفحص الروابط محليًا ويتحقق من سمعتها عبر
            <b>VirusTotal</b> عند تفعيل الخدمة، ثم يعرض تقييمًا أمنيًا موحدًا.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center; margin:8px 0 16px;">
            <h2 style="margin-bottom:4px;">👨‍💻 البرمجة والتطوير</h2>
            <div style="font-size:19px; font-weight:700; color:#087F8C;">
                Osama Almuhtadi
            </div>
            <div style="font-size:13px; color:#64748b; margin-top:4px;">
                مطوّر النظام
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Contact cards with recognizable official brand marks.
    st.markdown(
        """
        <style>
        .as-contact-grid {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:12px;
            direction:rtl;
        }
        .as-contact-card {
            display:flex; align-items:center; gap:12px;
            padding:14px 15px; border-radius:13px;
            border:1px solid rgba(148,163,184,.28);
            background:rgba(148,163,184,.07);
            min-height:66px; box-sizing:border-box;
        }
        .as-contact-icon {
            width:42px; height:42px; flex:0 0 42px;
            display:flex; align-items:center; justify-content:center;
            border-radius:12px; background:white;
        }
        .as-contact-title {font-size:12px; color:#64748b; margin-bottom:3px;}
        .as-contact-value {font-size:15px; font-weight:650; overflow-wrap:anywhere;}
        .as-contact-value a {text-decoration:none; color:inherit;}
        @media(max-width:520px) {
            .as-contact-grid {grid-template-columns:1fr;}
        }
        </style>

        <div class="as-contact-grid">
          <div class="as-contact-card">
            <div class="as-contact-icon">
              <svg viewBox="0 0 24 24" width="27" height="27" aria-label="Phone">
                <path fill="#16A34A" d="M6.6 10.8a15.5 15.5 0 006.6 6.6l2.2-2.2a1 1 0 011.02-.24c1.12.37 2.32.57 3.58.57a1 1 0 011 1V20a1 1 0 01-1 1C10.6 21 3 13.4 3 4a1 1 0 011-1h3.5a1 1 0 011 1c0 1.26.2 2.46.57 3.58a1 1 0 01-.25 1.02z"/>
              </svg>
            </div>
            <div>
              <div class="as-contact-title">الهاتف</div>
              <div class="as-contact-value" dir="ltr"><a href="tel:+967776998979">+967 776 998 979</a></div>
            </div>
          </div>

          <div class="as-contact-card">
            <div class="as-contact-icon">
              <svg viewBox="0 0 24 24" width="29" height="29" aria-label="Facebook">
                <path fill="#1877F2" d="M24 12.07C24 5.4 18.63 0 12 0S0 5.4 0 12.07C0 18.09 4.39 23.08 10.13 24v-8.44H7.08v-3.49h3.05V9.41c0-3.03 1.79-4.72 4.54-4.72 1.31 0 2.68.24 2.68.24v2.96h-1.51c-1.49 0-1.95.93-1.95 1.88v2.3h3.32l-.53 3.49h-2.79V24C19.61 23.08 24 18.09 24 12.07z"/>
              </svg>
            </div>
            <div>
              <div class="as-contact-title">فيسبوك</div>
              <div class="as-contact-value">اسامه محمد المهتدي</div>
            </div>
          </div>

          <div class="as-contact-card">
            <div class="as-contact-icon">
              <svg viewBox="0 0 24 24" width="29" height="29" aria-label="Instagram">
                <defs>
                  <linearGradient id="asInsta" x1="0" y1="1" x2="1" y2="0">
                    <stop offset="0%" stop-color="#FFB000"/>
                    <stop offset="48%" stop-color="#F00075"/>
                    <stop offset="100%" stop-color="#7B35C8"/>
                  </linearGradient>
                </defs>
                <rect x="2" y="2" width="20" height="20" rx="6" fill="none" stroke="url(#asInsta)" stroke-width="2.2"/>
                <circle cx="12" cy="12" r="4.3" fill="none" stroke="url(#asInsta)" stroke-width="2.2"/>
                <circle cx="18" cy="6.5" r="1.35" fill="#F00075"/>
              </svg>
            </div>
            <div>
              <div class="as-contact-title">إنستجرام</div>
              <div class="as-contact-value"><a href="https://www.instagram.com/oos7a/" target="_blank" rel="noopener">@oos7a</a></div>
            </div>
          </div>

          <div class="as-contact-card">
            <div class="as-contact-icon">
              <svg viewBox="0 0 24 24" width="29" height="29" aria-label="Telegram">
                <circle cx="12" cy="12" r="11" fill="#229ED9"/>
                <path fill="#fff" d="M18.9 6.8L16.4 18c-.19.79-.7.98-1.42.61l-3.93-2.9-1.9 1.83c-.21.21-.39.39-.8.39l.28-4 7.28-6.58c.32-.28-.07-.44-.49-.16L6.42 13l-3.83-1.2c-.83-.26-.85-.83.17-1.23l14.97-5.77c.69-.25 1.29.17 1.17 1.99z"/>
              </svg>
            </div>
            <div>
              <div class="as-contact-title">تليجرام</div>
              <div class="as-contact-value"><a href="https://t.me/oos7a" target="_blank" rel="noopener">@oos7a</a></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:12px;'>"
        "ArabShield AI Pro © 2026</div>",
        unsafe_allow_html=True
    )
