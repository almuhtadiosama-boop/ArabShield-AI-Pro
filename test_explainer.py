
from explainer import explain_conversation


conversation = """
المتصل: مبروك ربحت جائزة مالية كبيرة!
المستقبل: حقاً؟ كيف أستلم الجائزة؟
المتصل: أرسل رقم البطاقة والرقم السري فوراً.
المتصل: وحوّل رسوم استلام الجائزة الآن.
"""

result = explain_conversation(conversation)

print("نتيجة التحليل:")
print(result["result"])

print("احتمالية الاحتيال:", result["scam_probability"], "%")

print("الكلمات المؤثرة:")
for word, weight in result["important_words"]:
    print(word, round(weight, 4))
