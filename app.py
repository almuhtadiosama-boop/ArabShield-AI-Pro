
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

    st.header("حول ArabShield AI Pro")

    st.write("""
    ArabShield AI Pro هو نظام ذكي يهدف إلى اكتشاف
    مؤشرات الاحتيال الإلكتروني في المحادثات العربية
    باستخدام تقنيات التعلم العميق وشبكات LSTM.
    """)

    st.write("""
    يعتمد النظام على نموذج تعلم عميق لتحليل النصوص،
    ويستخدم تقنية LIME لتفسير الكلمات التي ساهمت
    في اتخاذ قرار التصنيف.
    """)

    st.write("""
    تم تطوير واجهة النظام باستخدام Streamlit،
    ويستخدم LIME لتقديم تفسير محلي لنتائج النموذج.
    """)
