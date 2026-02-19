# app/app.py
from __future__ import annotations

# ==============================================================================
# Imports
# ==============================================================================
import os, sys, re, json, datetime, smtplib, ssl, io
from email.message import EmailMessage
from datetime import datetime as dt, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from psycopg.types.json import Json  # psycopg v3 JSON adapter

# Optional mic recorder (pip install streamlit-mic-recorder)
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# ==============================================================================
# Path bootstrap
# ==============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.db import (
    session, fetch_one, fetch_dicts, exec_sql,
    ensure_patient_exists, create_user, authenticate_user,
    triage_load, triage_save,
)
from src.assistant.triage import TriageOrchestrator  # your orchestrator

# ==============================================================================
# App / LLM config
# ==============================================================================
load_dotenv()
st.set_page_config(page_title="Health AI", page_icon="🩺", layout="wide")

# Groq (multilingual)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_KEY   = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

USER_KEY = "auth_user"
IST = ZoneInfo("Asia/Kolkata")
def now_ist() -> dt: return dt.now(IST)
def ist_fmt(d: dt) -> str: return d.astimezone(IST).strftime("%a, %d %b %Y %I:%M %p IST")

# ------------------------------------------------------------------------------
# Language detection
# ------------------------------------------------------------------------------
def detect_lang(text: str) -> str:
    if any('\u0c00' <= ch <= '\u0c7f' for ch in text or ""):
        return "te"
    return "en"

# ------------------------------------------------------------------------------
# Symptom normalizer (for auto-reset + seeding)
# ------------------------------------------------------------------------------
def _norm_symptom(t: str) -> str | None:
    s = (t or "").lower()
    if "chest" in s and "pain" in s: return "chest pain"
    vocab = [
        "fever","cough","sore throat","diarrhea","vomiting","headache",
        "shortness of breath","abdominal pain","back pain","rash","fatigue"
    ]
    for k in vocab:
        if k in s: return k
    # aliases
    if "loose motion" in s or "loose stools" in s: return "diarrhea"
    if "belly pain" in s or "stomach pain" in s or "tummy pain" in s: return "abdominal pain"
    if "breathless" in s or ("short" in s and "breath" in s): return "shortness of breath"
    return None

# ------------------------------------------------------------------------------
# SMTP helpers
# ------------------------------------------------------------------------------
def _smtp_cfg():
    return {
        "host": os.getenv("SMTP_HOST", "").strip(),
        "port": int(os.getenv("SMTP_PORT", "0") or 0),
        "user": os.getenv("SMTP_USER", "").strip(),
        "pwd":  os.getenv("SMTP_PASS", os.getenv("SMTP_PASSWORD","")).strip(),
        "sender": (os.getenv("SMTP_FROM_NAME","Health AI").strip()
                   + f" <{os.getenv('SMTP_FROM', os.getenv('SMTP_USER','no-reply@example.com')).strip()}>"),
        "secure": (os.getenv("SMTP_SECURE", "starttls").strip().lower()),
        "override": [e.strip() for e in os.getenv("SMTP_TO_OVERRIDE", "").split(",") if e.strip()],
    }

def smtp_enabled() -> bool:
    c = _smtp_cfg()
    return bool(c["host"] and c["port"] and c["sender"])

def send_email(subject: str, body: str, recipients: list[str]) -> tuple[bool, str]:
    c = _smtp_cfg()
    if not smtp_enabled():
        return False, "SMTP not configured"
    to_list = c["override"] or recipients
    to_list = [r for r in to_list if r]
    if not to_list:
        return False, "no recipients"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = c["sender"]
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    try:
        if c["secure"] == "ssl":
            with smtplib.SMTP_SSL(c["host"], c["port"], context=ssl.create_default_context()) as s:
                if c["user"]: s.login(c["user"], c["pwd"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(c["host"], c["port"]) as s:
                if c["secure"] == "starttls":
                    s.starttls(context=ssl.create_default_context())
                if c["user"]: s.login(c["user"], c["pwd"])
                s.send_message(msg)
        return True, f"email sent to {', '.join(to_list)}"
    except Exception as e:
        return False, f"SMTP error: {e}"

def table_has_column(conn, schema_table: str, column: str) -> bool:
    schema, _, table = schema_table.partition(".")
    if not table:
        table, schema = schema, "public"
    row = fetch_one(conn, """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        LIMIT 1;
    """, (schema, table, column))
    return bool(row)

def fetch_user_basic(conn, user_id: str) -> dict | None:
    # Select only columns that exist
    base_cols = ["id", "email", "full_name"]
    extra_cols = []
    try:
        if table_has_column(conn, "app.users", "sex"):
            extra_cols.append("sex")
        if table_has_column(conn, "app.users", "age"):
            extra_cols.append("age")
    except Exception:
        pass
    cols = base_cols + extra_cols
    sql = f"SELECT {', '.join(cols)} FROM app.users WHERE id=%s LIMIT 1;"
    rows = fetch_dicts(conn, sql, (user_id,))
    row = rows[0] if rows else None
    if row is not None:
        row.setdefault("sex", None)
        row.setdefault("age", None)
    return row

def email_chat_reply(conn, user_id: str, question: str, reply_text: str, lang: str):
    user = fetch_user_basic(conn, user_id) or {}
    to = (user.get("email") or "").strip()
    if not to or not smtp_enabled(): return
    subject = "Your Health AI chat reply"
    body = (
        (f"Language: {'Telugu' if lang=='te' else 'English'}\n"
         f"Time (IST): {ist_fmt(now_ist())}\n\n"
         f"Your question:\n{question}\n\n"
         f"Recommendation:\n{reply_text}\n")
    )
    send_email(subject, body, [to])

def email_diet_plan(conn, user_id: str, targets: dict, day_plan: dict):
    user = fetch_user_basic(conn, user_id) or {}
    to = (user.get("email") or "").strip()
    if not to or not smtp_enabled(): return
    def block(k):
        b = day_plan.get(k, {})
        title = b.get("title", k.capitalize())
        tgt = b.get("target_kcal")
        lines = [f"{k.capitalize()} — {title}" + (f" · {tgt} kcal" if tgt else "")]
        for it in b.get("items", []):
            lines.append(f"- {it.get('item','Item')}: {it.get('portion','')}")
        return "\n".join(lines)
    subject = "Your Health AI diet plan"
    body = (
        f"Saved at: {ist_fmt(now_ist())}\n\n"
        f"Targets: {targets.get('kcal')} kcal | Protein {targets.get('protein_g')} g | "
        f"Carbs {targets.get('carbs_g')} g | Fat {targets.get('fat_g')} g\n\n"
        f"{block('breakfast')}\n\n{block('lunch')}\n\n{block('snack')}\n\n{block('dinner')}\n\n"
        f"Hydration: {day_plan.get('hydration',{}).get('notes','')}\n"
    )
    send_email(subject, body, [to])

# ==============================================================================
# Auth
# ==============================================================================
def auth_gate():
    st.title("Health AI")
    st.caption("Secure sign in to access your logs, diet, risk and chat.")
    t_login, t_signup = st.tabs(["Log in", "Sign up"])

    with t_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            pwd   = st.text_input("Password", type="password")
            ok = st.form_submit_button("Log in")
        if ok:
            try:
                with session(None) as conn:
                    user = authenticate_user(conn, email, pwd)
                if not user:
                    st.error("Invalid email or password.")
                else:
                    st.session_state[USER_KEY] = user
                    st.success(f"Welcome back, {user['full_name']}!")
                    st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with t_signup:
        with st.form("signup_form"):
            full_name = st.text_input("Full name")
            email2    = st.text_input("Email")
            pwd2      = st.text_input("Password", type="password", help="Min 8 chars.")
            ok2 = st.form_submit_button("Create account")
        if ok2:
            try:
                if len(pwd2.strip()) < 8:
                    st.warning("Password must be at least 8 characters.")
                else:
                    with session(None) as conn:
                        created = create_user(conn, email2, full_name, pwd2)
                    if not created:
                        st.warning("That email is already registered.")
                    else:
                        st.session_state[USER_KEY] = created
                        st.success("Account created!")
                        st.rerun()
            except Exception as e:
                st.error(f"Sign up failed: {e}")

# ==============================================================================
# DB helpers for issues / chat sessions
# ==============================================================================
def get_issue_col(conn) -> str:
    candidates = ["issue", "symptom", "name", "label", "title"]
    for c in candidates:
        if table_has_column(conn, "app.health_issues", c):
            return c
    row = fetch_one(conn, """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='app' AND table_name='health_issues'
        ORDER BY ordinal_position
        LIMIT 1;
    """)
    return row[0] if row and row[0] else "issue"

def ensure_chat_session(conn, user_id: str) -> str:
    row = fetch_one(
        conn,
        "SELECT id FROM app.chat_sessions WHERE patient_id=%s ORDER BY created_at DESC LIMIT 1;",
        (user_id,),
    )
    if row:
        return row[0]
    row = fetch_one(conn, """
        INSERT INTO app.chat_sessions (patient_id, title)
        VALUES (%s, 'General health chat')
        RETURNING id;
    """, (user_id,))
    return row[0]

def new_chat(conn, user_id: str) -> str:
    title = f"Chat {dt.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    row = fetch_one(conn, """
        INSERT INTO app.chat_sessions (patient_id, title)
        VALUES (%s, %s)
        RETURNING id;
    """, (user_id, title))
    return row[0]

def get_or_create_chat_session_id(conn, user_id: str) -> str:
    sid = st.session_state.get("chat_session_id")
    if sid:
        ok = fetch_one(conn,
            "SELECT 1 FROM app.chat_sessions WHERE id=%s AND patient_id=%s LIMIT 1;",
            (sid, user_id),
        )
        if not ok:
            st.session_state.pop("chat_session_id", None)
            sid = None
    if not sid:
        sid = ensure_chat_session(conn, user_id)
        st.session_state["chat_session_id"] = sid
    return sid

def recent_messages(conn, session_id: str, limit: int = 24):
    rows = fetch_dicts(conn, """
        SELECT sender, content
        FROM app.chat_messages
        WHERE session_id=%s
        ORDER BY created_at DESC
        LIMIT %s;
    """, (session_id, limit))
    return list(reversed(rows)) if rows else []

# ==============================================================================
# Issue extraction (rules fallback)
# ==============================================================================
SYMPTOM_LIST = [
    "fever","headache","cough","vomiting","diarrhea","chest pain",
    "shortness of breath","sore throat","rash","fatigue","back pain","abdominal pain"
]
DUR_RE  = re.compile(r"(?:for|since)\s+(?:the\s+last\s+)?(\d{1,3})\s*(day|days|week|weeks|month|months)", re.I)
DATE_RE = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})")

def _extract_issue_rules(text: str):
    t = text.lower()
    issue = next((s for s in SYMPTOM_LIST if s in t), None)
    days = None
    m = DUR_RE.search(t)
    if m:
        n = int(m.group(1)); unit = m.group(2).lower()
        mult = 1 if "day" in unit else (7 if "week" in unit else 30)
        days = n * mult
    onset = None
    m2 = DATE_RE.search(text)
    if m2:
        try:
            onset = datetime.date.fromisoformat(m2.group(1).replace("/", "-")).isoformat()
        except Exception:
            pass
    return {"issues":[{"issue":issue, "days":days, "onset_date":onset}]} if issue else {"issues":[]}

def normalize_issue_name(text: str) -> str:
    t = (text or "").strip().lower()
    if "chest" in t and "pain" in t:   return "chest pain"
    if "shortness of breath" in t:     return "shortness of breath"
    if "short" in t and "breath" in t: return "shortness of breath"
    SYM = {
        "fever": ["fever","feverish","high temp","temperature","getting fever"],
        "headache": ["headache","head ache","migraine","head pain"],
        "cough": ["cough","coughing"],
        "sore throat": ["sore throat","throat pain"],
        "vomiting": ["vomit","vomiting","threw up"],
        "diarrhea": ["diarrhea","loose stools","loose motions"],
        "rash": ["rash","skin rash"],
        "fatigue": ["fatigue","tiredness","exhaustion"],
        "back pain": ["back pain","backache","low back pain"],
        "abdominal pain": ["abdominal pain","stomach pain","belly pain","tummy pain"],
    }
    for canon, terms in SYM.items():
        if any(term in t for term in terms):
            return canon
    return " ".join(t.split()[:3]) or t

# ==============================================================================
# Recurrence + auto log
# ==============================================================================
def recent_issue_count(conn, user_id: str, canon_issue: str, days_window: int = 30) -> int:
    ic = get_issue_col(conn)
    row = fetch_one(conn, f"""
        SELECT count(*)
        FROM app.health_issues
        WHERE patient_id=%s
          AND lower({ic})=lower(%s)
          AND created_at >= now() - interval '%s days';
    """, (user_id, canon_issue, days_window))
    return int(row[0]) if row else 0

def recent_issue_dates(conn, user_id: str, canon_issue: str, days_window: int = 30) -> list[str]:
    ic = get_issue_col(conn)
    rows = fetch_dicts(conn, f"""
        SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS d
        FROM app.health_issues
        WHERE patient_id=%s
          AND lower({ic})=lower(%s)
          AND created_at >= now() - interval '%s days'
        GROUP BY 1
        ORDER BY 1 DESC;
    """, (user_id, canon_issue, days_window))
    return [r["d"] for r in (rows or [])]

def build_recurrence_tips(issue: str) -> str:
    tips = [
        "Keep a short daily log (timing, triggers, severity, meds).",
        "Hydrate and rest; gentle activity as tolerated.",
        "If symptoms worsen or red flags occur, seek urgent care.",
    ]
    return "- " + "\n".join(tips)

def auto_log_issue(conn, user_id: str, text: str) -> int:
    data = _extract_issue_rules(text)
    saved = 0
    ic = get_issue_col(conn)
    has_days   = table_has_column(conn, "app.health_issues", "days")
    has_onset  = table_has_column(conn, "app.health_issues", "onset_date")
    has_source = table_has_column(conn, "app.health_issues", "source")
    has_raw    = table_has_column(conn, "app.health_issues", "raw")
    has_alert  = table_has_column(conn, "app.health_issues", "alerted_at")

    for it in data.get("issues", []):
        raw_issue = (it.get("issue") or "").strip()
        if not raw_issue: continue
        issue = normalize_issue_name(raw_issue)
        days = it.get("days")
        onset_date = it.get("onset_date")

        dupe = fetch_one(conn, f"""
            SELECT 1 FROM app.health_issues
            WHERE patient_id=%s AND lower({ic})=lower(%s)
              AND created_at >= now() - interval '2 hours'
            LIMIT 1;
        """, (user_id, issue))
        if dupe: continue

        cols = ["patient_id", ic]; vals = [user_id, issue]
        if has_days:   cols.append("days");        vals.append(days)
        if has_onset:  cols.append("onset_date");  vals.append(onset_date)
        if has_source: cols.append("source");      vals.append("chat")
        if has_raw:    cols.append("raw");         vals.append({"raw": raw_issue, **it})
        cols_sql = ", ".join(cols); placeholders = ", ".join(["%s"] * len(vals))
        row = fetch_one(conn, f"""
            INSERT INTO app.health_issues ({cols_sql})
            VALUES ({placeholders})
            RETURNING id;
        """, tuple(vals))
        saved += 1; hi_id = row[0]

        try:
            cnt = recent_issue_count(conn, user_id, issue, days_window=30)
            if cnt >= 2:
                dates = recent_issue_dates(conn, user_id, issue, days_window=30)
                user = fetch_user_basic(conn, user_id) or {}
                to = (user.get("email") or "").strip()
                if to and smtp_enabled():
                    subject = f"Repeating {issue} in 30 days — please review"
                    body = (
                        f"Hello {user.get('full_name','')},\n\n"
                        f"Our system noticed repeating **{issue}** this month.\n"
                        f"Recent dates: {', '.join(dates) if dates else 'n/a'}\n\n"
                        f"Tips:\n{build_recurrence_tips(issue)}\n\n"
                        "If red-flag symptoms occur, seek urgent care immediately.\n"
                    )
                    send_email(subject, body, [to])
                if has_alert:
                    exec_sql(conn, "UPDATE app.health_issues SET alerted_at = now() WHERE id=%s;", (hi_id,))
        except Exception:
            pass

    return saved

# ==============================================================================
# Patient context (compact)
# ==============================================================================
def patient_context(conn, user_id: str) -> str:
    parts = []
    try:
        ra = fetch_dicts(conn, """
            SELECT created_at, band, score, rationale
            FROM app.risk_assessments
            WHERE patient_id=%s
            ORDER BY created_at DESC LIMIT 1;
        """, (user_id,))
        if ra:
            r = ra[0]; parts.append(f"RISK: band={r['band']}, score={r['score']} (at {r['created_at']})")
    except Exception:
        pass
    try:
        dp = fetch_dicts(conn, """
            SELECT created_at, targets, day_plan
            FROM app.diet_plans
            WHERE patient_id=%s
            ORDER BY created_at DESC LIMIT 1;
        """, (user_id,))
        if dp:
            t = dp[0]["targets"]
            parts.append(f"DIET: kcal={t.get('kcal')}, P={t.get('protein_g')}, C={t.get('carbs_g')}, F={t.get('fat_g')}")
    except Exception:
        pass
    try:
        ic = get_issue_col(conn)
        has_days = table_has_column(conn, "app.health_issues", "days")
        select_days = "days" if has_days else "NULL AS days"
        issues = fetch_dicts(conn, f"""
            SELECT {ic} AS label, {select_days}, onset_date
            FROM app.health_issues
            WHERE patient_id=%s
            ORDER BY created_at DESC
            LIMIT 5;
        """, (user_id,))
        if issues:
            def fmt(i):
                d = f" for {i.get('days')}d" if i.get('days') else ""
                o = f" (since {i.get('onset_date')})" if i.get('onset_date') else ""
                return f"{i['label']}{d}{o}"
            parts.append("ISSUES: " + "; ".join(map(fmt, issues)))
    except Exception:
        pass
    return "\n".join(parts)

# ==============================================================================
# Default chat mode
# ==============================================================================
if "mode_chat" not in st.session_state:
    st.session_state["mode_chat"] = "Guided triage"

# ==============================================================================
# MAIN
# ==============================================================================
user = st.session_state.get(USER_KEY)
if not user:
    auth_gate()
    st.stop()

user_id   = user["id"]
user_name = user["full_name"]

with session(user_id) as conn:
    ensure_patient_exists(conn, user_id)

st.markdown(f"### 👋 Hello **{user_name}** — how can I help?")
st.caption("Private to your account (RLS). Speak in Telugu or English, or type.")
st.divider()

# ---------------- TABS (UI only; NO chat_input inside) -------------------------
tab_home, tab_chat, tab_diet, tab_risk = st.tabs(["Home", "Chat (text/voice)", "Diet plan", "Risk scoring"])

# ----- Home -------------------------------------------------------------------
def render_how_it_works():
    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown(
            """
**Chat & LLM (Groq)** — Streaming chat (bilingual). We pass compact context + prior occurrences when relevant.

**Guided triage** — Short, symptom-specific follow-ups (max 2) → concise plan.

**Voice input** — Optional via `streamlit-mic-recorder`.

**Auto logging** — Issues mentioned in chat get saved to `app.health_issues`.

**Email** — Chat replies and diet plans can be mailed to your registered address.
            """
        )

with tab_home:
    render_how_it_works()
    with session(user_id) as conn:
        try:
            counts = fetch_dicts(conn, """
                SELECT 'diet_plans' AS tbl, count(*) FROM app.diet_plans WHERE patient_id=%s
                UNION ALL SELECT 'risk_assessments', count(*) FROM app.risk_assessments WHERE patient_id=%s
                UNION ALL SELECT 'health_issues', count(*) FROM app.health_issues WHERE patient_id=%s
                UNION ALL SELECT 'chat_messages', count(*) FROM app.chat_messages cm
                           JOIN app.chat_sessions cs ON cs.id = cm.session_id
                           WHERE cs.patient_id=%s
                ORDER BY tbl;
            """, (user_id,)*4)
            st.subheader("My snapshot")
            st.dataframe(pd.DataFrame(counts), use_container_width=True)
        except Exception as e:
            st.info("Snapshot will appear once you create some data (diet plan, risk, chat, etc.).")
            st.caption(f"(Details: {e})")

    if st.button("Sign out"):
        st.session_state.pop(USER_KEY, None)
        st.rerun()

# ----- Chat (tab UI only) -----------------------------------------------------
with tab_chat:
    render_how_it_works()
    with session(user_id) as conn:
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("🆕 New chat"):
                sid_new = new_chat(conn, user_id)
                st.session_state["chat_session_id"] = sid_new
                st.session_state.pop("orch", None)
                st.success("Started a new chat."); st.rerun()
        with c2:
            sid_tmp = get_or_create_chat_session_id(conn, user_id)
            st.session_state["chat_session_id"] = sid_tmp
            if st.button("🧹 Clear history"):
                exec_sql(conn, "DELETE FROM app.chat_messages WHERE session_id=%s;", (sid_tmp,))
                st.session_state.pop("orch", None)
                st.success("History cleared."); st.rerun()
        with c3:
            if st.button("🔧 Repair chat session"):
                st.session_state.pop("chat_session_id", None)
                st.success("Chat session repaired."); st.rerun()
            st.caption("Issues you mention (e.g., “3 రోజులుగా జ్వరం” / “fever for 3 days”) are auto-logged.")

        with session(user_id) as conn:
            sid = get_or_create_chat_session_id(conn, user_id)
            history = recent_messages(conn, sid)
        for m in history:
            with st.chat_message("user" if m["sender"] == "user" else "assistant"):
                st.markdown(m["content"])

        st.radio("Assistant mode", ["Guided triage", "General chat"],
                 horizontal=True, key="mode_chat")

        st.markdown("#### 🎙️ Voice input (Telugu/English)")
        if HAS_MIC:
            if "voice_state" not in st.session_state:
                st.session_state.voice_state = {"text": "", "ready": False, "lang_hint": "auto"}

            colv1, colv2 = st.columns([2,1])
            with colv1:
                st.caption("Click once to start, click again to stop. Works best in Chrome.")
                audio = mic_recorder(
                    key="mic",
                    start_prompt="Start recording",
                    stop_prompt="Stop recording",
                    just_once=True
                )
            with colv2:
                lang_choice = st.selectbox("Speech language (hint)", ["auto","Telugu","English"])
                st.session_state.voice_state["lang_hint"] = "te" if lang_choice=="Telugu" else ("en" if lang_choice=="English" else "auto")

            if audio and audio.get("bytes") and not st.session_state.voice_state["ready"]:
                # local stub (no external transcription in this file)
                st.session_state.voice_state["text"] = ""
                st.session_state.voice_state["ready"] = True
                st.rerun()

            if st.session_state.voice_state["ready"]:
                st.success("Transcription complete.")
                st.text_area("Transcript", value=st.session_state.voice_state["text"], height=100, key="voice_text_preview")
                cva, cvb, _ = st.columns([1,1,2])
                with cva:
                    if st.button("Send transcript"):
                        st.session_state["pending_voice_to_send"] = st.session_state.voice_state["text"]
                        st.session_state.voice_state = {"text": "", "ready": False, "lang_hint": st.session_state.voice_state["lang_hint"]}
                        st.rerun()
                with cvb:
                    if st.button("Discard"):
                        st.session_state.voice_state = {"text": "", "ready": False, "lang_hint": st.session_state.voice_state["lang_hint"]}
                        st.rerun()
        else:
            st.info("Install voice support: `pip install streamlit-mic-recorder`")

# ----- Diet plan (simple stub) -------------------------------------------------
with tab_diet:
    st.subheader("Diet plan (LLM-like, local stub)")
    st.caption("Compute energy target and store a simple plan (no external calls here).")

    sex      = st.selectbox("Sex", ["female", "male"])
    age      = st.number_input("Age (years)", min_value=10, max_value=100, value=30)
    weight   = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    height   = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
    activity = st.selectbox("Activity", ["sedentary", "light", "moderate", "active"])

    if st.button("Save sample diet plan"):
        bmr  = (10 * weight) + (6.25 * height) - (5 * age) + (5 if sex == "male" else -161)
        mult = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}[activity]
        kcal_target = int(round(bmr * mult))
        targets = {
            "kcal": kcal_target,
            "protein_g": int(round(0.8 * weight)),
            "carbs_g": int(round(0.50 * kcal_target / 4)),
            "fat_g": int(round(0.30 * kcal_target / 9))
        }
        plan = {
            "breakfast":{"title":"Simple Indian","target_kcal":450,"items":[{"item":"Idli + sambar","portion":"3 + 1 cup"}]},
            "lunch":{"title":"Rice + dal","target_kcal":650,"items":[{"item":"Rice + dal + veg","portion":"1.5 cups + 1 cup"}]},
            "snack":{"title":"Fruit + curd","target_kcal":250,"items":[{"item":"Banana + curd","portion":"1 + 1/2 cup"}]},
            "dinner":{"title":"Chapati + sabzi","target_kcal":650,"items":[{"item":"Chapati + veg curry","portion":"3 + 1 cup"}]},
            "hydration":{"notes":"Sip water/ORS through the day"},
        }
        with session(user_id) as conn:
            exec_sql(conn, """
                INSERT INTO app.diet_plans (patient_id, inputs, targets, day_plan)
                VALUES (%s, %s, %s, %s);
            """, (user_id, Json({"sex":sex,"age":age,"weight":weight,"height":height,"activity":activity}),
                  Json(targets), Json(plan)))
            email_diet_plan(conn, user_id, targets, plan)
        st.success(f"Saved diet plan ({targets['kcal']} kcal) — emailed to you.")

# ----- Risk scoring ------------------------------------------------------------
with tab_risk:
    st.subheader("Risk scoring (MVP)")
    st.caption("Simple additive score — not a diagnosis.")

    fever      = st.checkbox("High fever (≥ 39.5°C)")
    chest_pain = st.checkbox("Chest pain")
    breath     = st.checkbox("Shortness of breath")
    severe     = st.checkbox("Severe bleeding / trauma")
    chronic    = st.checkbox("Significant chronic disease")

    if st.button("Compute & save risk"):
        score = 0
        score += 2 if fever else 0
        score += 3 if chest_pain else 0
        score += 3 if breath else 0
        score += 3 if severe else 0
        score += 1 if chronic else 0
        band = "low"
        if score >= 6: band = "high"
        elif score >= 3: band = "medium"
        rationale = "no notable risk factors" if score == 0 else "symptom flags present"
        try:
            with session(user_id) as conn:
                exec_sql(conn, """
                    INSERT INTO app.risk_assessments (patient_id, band, score, rationale)
                    VALUES (%s, %s, %s, %s);
                """, (user_id, band, score, rationale))
            st.success(f"Saved risk assessment → band: {band}, score: {score}")
        except Exception as e:
            st.error(f"Save failed: {e}")

# ==============================================================================
# ROOT-LEVEL chat input (outside any tab/expander) — works for all tabs
# ==============================================================================
with session(user_id) as conn:
    sid = get_or_create_chat_session_id(conn, user_id)

typed_text = st.chat_input("Type here… (works for all tabs)", key="chat_input", max_chars=2000)
voice_once = st.session_state.pop("pending_voice_to_send", None)
incoming_msg = (voice_once or typed_text or "").strip()
if not incoming_msg:
    st.stop()

msg_text = incoming_msg
lang = detect_lang(msg_text)

# 1) Save user message
try:
    with session(user_id) as conn:
        sid = get_or_create_chat_session_id(conn, user_id)
        exec_sql(conn, "INSERT INTO app.chat_messages (session_id, sender, content) VALUES (%s,'user',%s);",
                 (sid, msg_text))
except Exception as e:
    st.error(f"Chat failed (save user message): {e}")
    st.stop()

mode_now = st.session_state.get("mode_chat", "Guided triage")

# ------------------- GUIDED TRIAGE --------------------------------------------
if mode_now == "Guided triage":
    with session(user_id) as conn:
        sid = get_or_create_chat_session_id(conn, user_id)
        tri_state = triage_load(conn, user_id, sid) or {}

    # demographics if present (to avoid irrelevant questions)
    user_row = {}
    try:
        with session(user_id) as conn:
            user_row = fetch_user_basic(conn, user_id) or {}
    except Exception:
        pass
    sex = (user_row.get("sex") or "").lower() if user_row else None
    age = user_row.get("age") if user_row else None

    orch: TriageOrchestrator = st.session_state.get("orch")
    if not orch:
        orch = TriageOrchestrator(groq_client, GROQ_MODEL, lang, sex=sex or None, age=age)
        if tri_state:
            orch.slots.update(tri_state.get("slots", {}))
        st.session_state["orch"] = orch

    # ---- AUTO-RESET LOGIC ----
    try:
        last_completed = bool(tri_state.get("completed")) or bool(orch.slots.get("completed"))
    except Exception:
        last_completed = False

    new_sym = _norm_symptom(msg_text)
    old_sym = orch.slots.get("symptom")

    if last_completed or (new_sym and old_sym and new_sym != old_sym):
        orch.slots = {}
        st.session_state["orch"] = orch
        with session(user_id) as conn:
            sid = get_or_create_chat_session_id(conn, user_id)
            triage_save(conn, user_id, sid, {"slots": {}}, False)

    # If this is a new chat and the first message contains a symptom, seed it.
    if new_sym and not orch.slots.get("symptom"):
        orch.slots["symptom"] = new_sym

    # ingest and ask next
    orch.ingest_user_answer(msg_text)

    with session(user_id) as conn:
        sid = get_or_create_chat_session_id(conn, user_id)
        hist_text = "\n".join([m["content"] for m in recent_messages(conn, sid)])

    q, done = orch.propose_next_question(hist_text)

    if not done:
        with session(user_id) as conn:
            sid = get_or_create_chat_session_id(conn, user_id)
            triage_save(conn, user_id, sid, {"slots": orch.slots}, False)
            exec_sql(conn, "INSERT INTO app.chat_messages (session_id, sender, content) VALUES (%s,'bot',%s);",
                     (sid, q))
        st.rerun()

    # done → produce final summary + diet, mark completed, save, email
    with session(user_id) as conn:
        ctx = patient_context(conn, user_id)
    try:
        summary = orch.generate_summary(ctx)
    except Exception as e:
        summary = ("### సారాంశం\n- క్షమించండి—లోపం: " + str(e)) if lang=="te" else f"### Summary\n- Sorry—error: {e}"

    diet = orch.generate_diet_json()
    try:
        with session(user_id) as conn:
            sid = get_or_create_chat_session_id(conn, user_id)
            exec_sql(conn, """
                INSERT INTO app.diet_plans (patient_id, inputs, targets, day_plan)
                VALUES (%s, %s, %s, %s);
            """, (user_id, Json({"from":"triage","slots": orch.slots}), Json(diet["targets"]), Json(diet["day_plan"])))
            email_chat_reply(conn, user_id, "Guided triage summary + diet", summary, lang)
            email_diet_plan(conn, user_id, diet["targets"], diet["day_plan"])

            orch.slots["completed"] = True  # mark triage complete
            triage_save(conn, user_id, sid, {"slots": orch.slots}, True)

            exec_sql(conn, "INSERT INTO app.chat_messages (session_id, sender, content) VALUES (%s,'bot',%s);",
                     (sid, summary))
    except Exception as e:
        st.warning(f"Saved with warnings: {e}")
    st.rerun()

# ------------------- GENERAL CHAT (safe fallback) -----------------------------
try:
    with session(user_id) as conn:
        sid = get_or_create_chat_session_id(conn, user_id)
        _ = auto_log_issue(conn, user_id, msg_text)  # best-effort log
        ctx = patient_context(conn, user_id)

    reply_text = (
        "### సారాంశం\n- మీ సందేశం అందింది; సమాచారం పరిమితంగా ఉంది.\n\n"
        "### ఇప్పుడే చేసుకోవాల్సినవి\n- నీరు/విశ్రాంతి; OTC లేబల్ ప్రకారం మాత్రమే.\n\n"
        "### ఎప్పుడు అత్యవసరం\n- తీవ్రమైన/మరుగుదల లక్షణాలు.\n\n"
        "### తర్వాతి అడుగులు\n- వివరాలు మరింత ఇవ్వండి లేదా గైడెడ్ ట్రయాజ్ ప్రయత్నించండి."
        if lang=="te" else
        "### Summary\n- Got your message; details are limited.\n\n"
        "### Self-care now\n- Hydrate, rest; OTC per label.\n\n"
        "### When to seek urgent care\n- Severe/worsening symptoms.\n\n"
        "### Next steps\n- Provide more detail or switch to Guided triage."
    )

    with session(user_id) as conn:
        sid = get_or_create_chat_session_id(conn, user_id)
        exec_sql(conn, "INSERT INTO app.chat_messages (session_id, sender, content) VALUES (%s, 'bot', %s);",
                 (sid, reply_text))
        email_chat_reply(conn, user_id, msg_text, reply_text, lang)
except Exception as e:
    st.error(f"Chat failed (save/email bot reply): {e}")

st.rerun()
