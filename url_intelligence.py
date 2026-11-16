"""
ArabShield URL Intelligence V2
Local URL structure and brand-impersonation heuristics.

Important:
- This is a heuristic screening module, not a live reputation service.
- A risk score is not a probability of fraud.
- Keep the official-domain allowlist reviewed and maintained.
"""
import ipaddress
import re
from urllib.parse import urlsplit, unquote

URL_RE = re.compile(r"""(?ix)\b(?:https?://|www\.)[^\s<>"'`]+""")

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly",
    "rb.gy", "shorturl.at", "ow.ly", "rebrand.ly", "buff.ly"
}

SUSPICIOUS_TLDS = {
    "top", "xyz", "click", "work", "support", "gq", "tk",
    "ml", "cf", "ga", "buzz", "rest", "fit", "online"
}

# Keep this allowlist limited to domains confirmed from the brand's official channels.
OFFICIAL_DOMAINS = {
    "yemenmobile": {"yemenmobile.com.ye"},
}

BRAND_ALIASES = {
    "yemenmobile": ("yemenmobile", "yemen-mobile", "يمن موبايل"),
    "you": ("you", "you.com.ye"),
    "sabafon": ("sabafon", "سبأفون", "سبافون"),
    "cacbank": ("cacbank", "كاك بنك"),
    "alrajhi": ("alrajhi", "الراجحي"),
    "paypal": ("paypal",),
    "microsoft": ("microsoft",),
    "google": ("google",),
    "facebook": ("facebook",),
    "whatsapp": ("whatsapp",),
    "telegram": ("telegram",),
    "apple": ("apple",),
}

def _compact(value):
    """Lowercase and remove separators for tolerant brand matching."""
    value = unquote(str(value or "")).lower()
    value = re.sub(r"[\s._\-]+", "", value)
    return value

def extract_urls(text):
    """Extract URL-like strings and trim common trailing punctuation."""
    found = []
    for match in URL_RE.findall(str(text or "")):
        cleaned = match.rstrip(".,،؛:!?؟)]}>\"'")
        if cleaned.lower().startswith("www."):
            cleaned = "http://" + cleaned
        if cleaned and cleaned not in found:
            found.append(cleaned)
    return found

def _official_match(host, brand_key):
    """Only accept the exact official host or a true subdomain of it."""
    for official in OFFICIAL_DOMAINS.get(brand_key, set()):
        if host == official or host.endswith("." + official):
            return True
    return False

def _brand_mentions(value):
    compact = _compact(value)
    hits = []
    for brand_key, aliases in BRAND_ALIASES.items():
        if any(_compact(alias) in compact for alias in aliases):
            hits.append(brand_key)
    return hits

def analyze_url(url):
    """Return transparent heuristic indicators for one URL."""
    raw = str(url or "").strip()
    candidate = raw if "://" in raw else "http://" + raw
    flags = []
    score = 0
    brand_findings = []

    try:
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            return {
                "url": raw, "host": "", "score": 0,
                "level": "غير قابل للتحليل",
                "flags": ["تعذر استخراج اسم النطاق"],
                "brand_findings": []
            }

        try:
            ipaddress.ip_address(host)
            flags.append("يستخدم عنوان IP مباشرة بدل اسم نطاق")
            score += 25
        except ValueError:
            pass

        if parts.username or parts.password:
            flags.append("يحتوي الرابط على بيانات اعتماد قبل اسم النطاق")
            score += 25

        if "xn--" in host:
            flags.append("يحتوي النطاق على ترميز Punycode؛ افحص احتمال تشابه الحروف")
            score += 20

        if host in SHORTENERS:
            flags.append("يستخدم خدمة اختصار روابط؛ الوجهة النهائية غير ظاهرة")
            score += 15

        labels = host.split(".")
        if len(labels) >= 5:
            flags.append("يحتوي على عدد كبير من مستويات النطاق الفرعية")
            score += 10

        tld = labels[-1] if len(labels) > 1 else ""
        if tld in SUSPICIOUS_TLDS:
            flags.append(f"امتداد النطاق ({tld}) يستحق فحصًا إضافيًا؛ ليس دليلًا وحده")
            score += 10

        if parts.scheme.lower() != "https":
            flags.append("الاتصال لا يستخدم HTTPS")
            score += 5

        if parts.port and parts.port not in (80, 443):
            flags.append(f"يستخدم منفذًا غير شائع ({parts.port})")
            score += 10

        # Check the full URL, not only the hostname, for brand references.
        full_reference = " ".join([host, parts.path, parts.query, parts.fragment])
        mentioned = _brand_mentions(full_reference)

        for brand_key in mentioned:
            official = _official_match(host, brand_key)
            if official:
                brand_findings.append({
                    "brand": brand_key,
                    "status": "official_domain_match",
                    "message": "النطاق يطابق النطاق الرسمي المدرج لهذه العلامة"
                })
            else:
                brand_findings.append({
                    "brand": brand_key,
                    "status": "possible_impersonation",
                    "message": "ظهر اسم العلامة داخل الرابط لكن النطاق لا يطابق النطاق الرسمي المدرج"
                })
                flags.append(
                    f"اشتباه انتحال علامة تجارية: ظهر اسم {brand_key} "
                    "في الرابط دون تطابق النطاق مع القائمة الرسمية"
                )
                score += 40

        decoded = unquote(raw)
        if len(raw) > 180:
            flags.append("الرابط طويل بشكل ملحوظ")
            score += 5
        if re.search(r"%[0-9a-fA-F]{2}", raw):
            flags.append("يحتوي على ترميز URL؛ افحص الوجهة والمعاملات")
            score += 5
        if "@" in parts.netloc:
            flags.append("يحتوي جزء المضيف على الرمز @ وقد يضلل بشأن الوجهة")
            score += 20
        if re.search(r"(login|verify|gift|prize|bonus|claim|free|update|secure)", decoded, re.I):
            flags.append("يحتوي مسار الرابط أو معاملاته على كلمات استدراج شائعة")
            score += 5

        score = min(score, 100)
        if score >= 50:
            level = "مرتفع"
        elif score >= 20:
            level = "متوسط"
        else:
            level = "منخفض"

        return {
            "url": raw,
            "host": host,
            "score": score,
            "level": level,
            "flags": flags or ["لم تظهر مؤشرات شكلية واضحة في الفحص المحلي"],
            "brand_findings": brand_findings
        }

    except Exception:
        return {
            "url": raw, "host": "", "score": 0,
            "level": "غير قابل للتحليل",
            "flags": ["حدث خطأ أثناء تحليل بنية الرابط"],
            "brand_findings": []
        }

def analyze_text_urls(text):
    """Extract and analyze all URLs in a message."""
    return [analyze_url(url) for url in extract_urls(text)]


def check_virustotal_url(url, api_key, timeout=15):
    """
    Query an existing VirusTotal URL report.
    If no report exists, submit the URL for analysis and return its analysis ID.
    Submitted/queried URLs may be added to VirusTotal's shared dataset.
    """
    import base64
    import requests

    headers = {"x-apikey": api_key, "accept": "application/json"}
    url_id = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")

    try:
        report_response = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
            timeout=timeout
        )

        if report_response.status_code == 200:
            payload = report_response.json()
            attrs = payload.get("data", {}).get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            return {
                "status": "report",
                "stats": stats,
                "reputation": attrs.get("reputation"),
                "last_analysis_date": attrs.get("last_analysis_date"),
                "permalink": f"https://www.virustotal.com/gui/url/{url_id}",
                "message": "تم العثور على تقرير سابق في VirusTotal."
            }

        if report_response.status_code == 404:
            submit_response = requests.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=headers,
                data={"url": url},
                timeout=timeout
            )
            submit_response.raise_for_status()
            analysis_id = submit_response.json().get("data", {}).get("id")
            return {
                "status": "submitted",
                "analysis_id": analysis_id,
                "permalink": f"https://www.virustotal.com/gui/url/{url_id}",
                "message": "لم يوجد تقرير سابق؛ أُرسل الرابط للفحص. أعد المحاولة لاحقًا لعرض النتيجة."
            }

        if report_response.status_code == 429:
            return {
                "status": "limited",
                "message": "تم بلوغ حد طلبات VirusTotal مؤقتًا."
            }

        report_response.raise_for_status()

    except requests.exceptions.Timeout:
        return {"status": "error", "message": "انتهت مهلة الاتصال بخدمة VirusTotal."}
    except requests.exceptions.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 401:
            message = "مفتاح VirusTotal غير صحيح أو غير مصرح به."
        elif status_code == 403:
            message = "حساب API لا يملك صلاحية هذا الطلب."
        elif status_code == 429:
            message = "تم بلوغ حد طلبات VirusTotal."
        else:
            message = f"تعذر الاتصال بـVirusTotal: {exc}"
        return {"status": "error", "message": message}
    except Exception:
        return {"status": "error", "message": "حدث خطأ غير متوقع أثناء فحص VirusTotal."}


def calculate_unified_risk(lstm_probability, url_results, vt_results):
    """
    Combine available signals into a transparent triage score.
    The output is a heuristic risk indicator, NOT a calibrated probability.
    """
    lstm_score = max(0.0, min(100.0, float(lstm_probability) * 100.0))

    local_scores = [float(item.get("score", 0)) for item in (url_results or [])]
    local_score = max(local_scores) if local_scores else None

    vt_malicious = 0
    vt_suspicious = 0
    vt_has_report = False
    for item in (vt_results or []):
        if item.get("status") == "report":
            stats = item.get("stats", {})
            vt_malicious += int(stats.get("malicious", 0) or 0)
            vt_suspicious += int(stats.get("suspicious", 0) or 0)
            vt_has_report = True

    vt_score = None
    if vt_has_report:
        if vt_malicious >= 5:
            vt_score = 95.0
        elif vt_malicious >= 2:
            vt_score = 80.0
        elif vt_malicious == 1:
            vt_score = 65.0
        elif vt_suspicious >= 3:
            vt_score = 65.0
        elif vt_suspicious > 0:
            vt_score = 50.0
        else:
            # No detections is not proof of safety.
            vt_score = 15.0

    components = [
        ("تحليل LSTM", lstm_score, 0.45),
    ]
    if local_score is not None:
        components.append(("تحليل الرابط المحلي", local_score, 0.25))
    if vt_score is not None:
        components.append(("سمعة VirusTotal", vt_score, 0.30))

    weight_total = sum(weight for _, _, weight in components)
    combined = sum(score * weight for _, score, weight in components) / weight_total

    # Conservative floors for corroborated external detections.
    if vt_malicious >= 2:
        combined = max(combined, 70.0)
    elif vt_malicious == 1:
        combined = max(combined, 55.0)
    if local_score is not None and local_score >= 50 and lstm_score >= 50:
        combined = max(combined, 65.0)

    if combined >= 65:
        level = "مرتفع"
    elif combined >= 35:
        level = "متوسط"
    else:
        level = "منخفض"

    reasons = []
    if lstm_score >= 50:
        reasons.append("نموذج LSTM أعطى مؤشرًا لاحتمال الاحتيال")
    if local_score is not None and local_score >= 20:
        reasons.append("تحليل الرابط المحلي رصد مؤشرات تستحق الانتباه")
    if vt_malicious:
        reasons.append(f"VirusTotal: {vt_malicious} محرك/محركات صنفت الرابط ضارًا")
    if vt_suspicious:
        reasons.append(f"VirusTotal: {vt_suspicious} محرك/محركات صنفت الرابط مشبوهًا")
    if vt_has_report and vt_malicious == 0 and vt_suspicious == 0:
        reasons.append("لم يرصد تقرير VirusTotal الحالي محركات ضارة؛ وهذا لا يثبت السلامة")
    if not reasons:
        reasons.append("لم تظهر مؤشرات قوية ضمن الفحوص المتاحة")

    return {
        "score": round(combined, 1),
        "level": level,
        "lstm_score": round(lstm_score, 1),
        "local_score": round(local_score, 1) if local_score is not None else None,
        "virustotal_score": round(vt_score, 1) if vt_score is not None else None,
        "vt_malicious": vt_malicious,
        "vt_suspicious": vt_suspicious,
        "components": components,
        "reasons": reasons,
        "is_probability": False,
    }
