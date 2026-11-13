# -*- coding: utf-8 -*-
# app/app.py – UPDATED: Chronological messages, auto-email after recommendations
from __future__ import annotations
import os, sys, importlib, importlib.util, tempfile, time, json, re
from datetime import datetime as dt
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------- cookies ----------
try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    class EncryptedCookieManager:
        def __init__(self, prefix: str = "", password: str = ""):
            self._prefix = prefix
            if "_st_cookie_store" not in st.session_state:
                st.session_state["_st_cookie_store"] = {}
            self._store = st.session_state["_st_cookie_store"]
        def ready(self) -> bool: return True
        def get(self, key: str, default: str = "") -> str:
            return self._store.get(self._prefix + key, default)
        def __setitem__(self, key: str, value: str):
            self._store[self._prefix + key] = value
        def save(self): pass

# ---------- mic dependency ----------
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# ---------- module loading ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "src"),
          os.path.join(PROJECT_ROOT, "src", "core"),
          os.path.join(PROJECT_ROOT, "src", "assistant")):
    if p not in sys.path:
        sys.path.insert(0, p)

def _load_module_by_path(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_ENHANCED_DB_PATH = os.path.join(PROJECT_ROOT, "src", "core", "enhanced_db.py")
try:
    enhanced_db = importlib.import_module("enhanced_db")
    if not hasattr(enhanced_db, "session"):
        raise AttributeError("loaded enhanced_db has no 'session'")
except Exception:
    try:
        enhanced_db = importlib.import_module("core.enhanced_db")
        if not hasattr(enhanced_db, "session"):
            raise AttributeError("loaded core.enhanced_db has no 'session'")
    except Exception:
        enhanced_db = _load_module_by_path("enhanced_db_forced", _ENHANCED_DB_PATH)

intelligent_assistant = importlib.import_module("intelligent_assistant")
email_service_mod = importlib.import_module("email_service")
HealthAssistant = intelligent_assistant.HealthAssistant
get_email_service = email_service_mod.get_email_service

# ---------- config ----------
load_dotenv()
st.set_page_config(page_title="Health AI Assistant", page_icon="🥼", layout="wide")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
if not groq_client:
    st.error("❌ GROQ_API_KEY not found in .env"); st.stop()

cookies = EncryptedCookieManager(prefix="healthai_", password=os.getenv("COOKIE_SECRET", "please-change-me"))
if not cookies.ready(): st.stop()
AUTH_COOKIE = "session_token"

IST = ZoneInfo("Asia/Kolkata")
def now_ist() -> dt: return dt.now(IST)
def ist_fmt(d: dt) -> str: return d.astimezone(IST).strftime("%a, %d %b %Y %I:%M %p IST")
def detect_lang(text: str) -> str: return "te" if any('\u0c00' <= ch <= '\u0c7f' for ch in (text or "")) else "en"

USER_KEY = "auth_user"
SESSION_DAYS = 30

# ---------- voice transcription ----------
def transcribe_audio(audio_bytes: bytes, language: str = "auto") -> str:
    """Transcribe audio using Groq Whisper API"""
    if not audio_bytes:
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language=None if language == "auto" else language,
                response_format="text",
                temperature=0.0,
            )
        
        os.unlink(tmp_path)
        result = transcription.strip() if transcription else ""
        return result
        
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return ""

# ---------- auth ----------
def auth_gate():
    st.title("🥼 Health AI Assistant")
    st.caption("Your intelligent health companion – English & Telugu")

    tok = cookies.get(AUTH_COOKIE)
    if (USER_KEY not in st.session_state) and tok:
        try:
            with enhanced_db.session(None) as conn:
                u = enhanced_db.get_user_by_session_token(conn, tok)
            if u: 
                st.session_state[USER_KEY] = u
                cookies[AUTH_COOKIE] = tok
                cookies.save()
        except Exception:
            cookies[AUTH_COOKIE] = ""
            cookies.save()
    
    if USER_KEY in st.session_state: 
        return True

    tab_login, tab_signup = st.tabs(["🔒 Sign In", "✨ Create Account"])
    
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            pwd = st.text_input("Password", type="password")
            remember = st.checkbox("Keep me signed in (30 days)", value=True)
            if st.form_submit_button("Sign In", use_container_width=True):
                if not email or not pwd:
                    st.error("Please enter both email and password")
                else:
                    try:
                        with enhanced_db.session(None) as conn:
                            user = enhanced_db.authenticate_user(conn, email, pwd)
                        if not user:
                            st.error("❌ Invalid email or password")
                        else:
                            days = SESSION_DAYS if remember else 1
                            with enhanced_db.session(None) as conn:
                                tok = enhanced_db.create_auth_session(conn, user["id"], days=days)
                            cookies[AUTH_COOKIE] = tok
                            cookies.save()
                            st.session_state[USER_KEY] = user
                            st.success(f"✅ Welcome back, {user['full_name']}!")
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Login error: {e}")

    with tab_signup:
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name *", placeholder="John Doe")
                email = st.text_input("Email *", placeholder="john@example.com")
                pwd = st.text_input("Password *", type="password")
            with col2:
                guardian_name = st.text_input("Guardian Name (Optional)")
                guardian_email = st.text_input("Guardian Email (Optional)")
                phone = st.text_input("Phone (Optional)", placeholder="+91 9876543210")
            c3, c4 = st.columns(2)
            with c3: age = st.number_input("Age", min_value=1, max_value=120, value=30)
            with c4: sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            if st.form_submit_button("Create Account", use_container_width=True):
                if not all([full_name, email, pwd]):
                    st.error("Please fill all required fields (*)")
                elif len(pwd) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        with enhanced_db.session(None) as conn:
                            user = enhanced_db.create_user(
                                conn, email, full_name, pwd,
                                guardian_email=guardian_email or None,
                                guardian_name=guardian_name or None,
                                phone=phone or None, age=age, sex=sex.lower()
                            )
                        if not user:
                            st.error("❌ Email already registered")
                        else:
                            with enhanced_db.session(None) as conn:
                                tok = enhanced_db.create_auth_session(conn, user["id"], days=SESSION_DAYS)
                            cookies[AUTH_COOKIE] = tok
                            cookies.save()
                            st.session_state[USER_KEY] = user
                            st.success("✅ Account created successfully!")
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error creating account: {e}")
    
    return USER_KEY in st.session_state

def signout_button():
    if st.button("🚪 Sign Out", use_container_width=True):
        tok = cookies.get(AUTH_COOKIE)
        if tok:
            try:
                with enhanced_db.session(None) as conn: 
                    enhanced_db.revoke_session(conn, tok)
            except Exception:
                pass
            cookies[AUTH_COOKIE] = ""
            cookies.save()
        st.session_state.clear()
        st.rerun()

# ---------- assistant ----------
def get_assistant(lang: str):
    if "assistant" not in st.session_state or st.session_state.get("assistant_lang") != lang:
        st.session_state["assistant"] = HealthAssistant(
            groq_client=groq_client, 
            model=GROQ_MODEL, 
            lang=lang
        )
        st.session_state["assistant_lang"] = lang
    return st.session_state["assistant"]

def get_consultation_language(user_id: str) -> str:
    # Check if user has selected language
    if "user_selected_lang" in st.session_state:
        return st.session_state["user_selected_lang"]
    
    session_id = st.session_state.get("consultation_id")
    if not session_id:
        return st.session_state.get("user_selected_lang", "en")
    
    if "consultation_lang" in st.session_state:
        return st.session_state["consultation_lang"]
    
    with enhanced_db.session(user_id) as conn:
        row = enhanced_db.fetch_one(conn, """
            SELECT content FROM app.conversation_history
            WHERE session_id=%s AND role='user'
            ORDER BY created_at DESC LIMIT 1;
        """, (session_id,))
        last_user_text = row[0] if row else ""
        detected = detect_lang(last_user_text)
        st.session_state["consultation_lang"] = detected
        return detected

def send_consultation_email(user_id: str, summary_text: str):
    """Send email with consultation summary"""
    user = st.session_state.get(USER_KEY)
    session_id = st.session_state.get("consultation_id")
    
    if not user or not session_id:
        return False, "Missing user or session data"
    
    lang = get_consultation_language(user_id)
    email_service = get_email_service()
    
    if not email_service.is_enabled():
        return False, "Email not configured"
    
    try:
        success, msg, recipients = email_service.send_final_recommendation(
            patient_name=user["full_name"],
            patient_email=user["email"],
            guardian_email=user.get("guardian_email"),
            recommendation_text=summary_text,
            consultation_date=now_ist(),
            lang=lang,
        )
        
        # Log to database
        with enhanced_db.session(user_id) as conn:
            enhanced_db.log_email_to_db(
                conn,
                patient_id=user_id,
                session_id=session_id,
                email_type="final_recommendation_auto",
                recipients=recipients if success else [],
                subject=f"{'Health Consultation Summary' if lang=='en' else 'ఆరోగ్య సలహా సారాంశం'} - {now_ist().strftime('%B %d, %Y')}",
                status="sent" if success else "failed"
            )
        
        return success, msg if not success else f"Sent to: {', '.join(recipients)}"
        
    except Exception as e:
        return False, str(e)

# ---------- history ----------
def get_session_lang(conn, session_id: str) -> str:
    row = enhanced_db.fetch_one(conn, """
        SELECT content FROM app.conversation_history
        WHERE session_id=%s AND role='user'
        ORDER BY created_at DESC LIMIT 1;
    """, (session_id,))
    last_user_text = row[0] if row else ""
    return detect_lang(last_user_text) or "en"

def render_chat_messages(conn, session_id: str, chronological: bool = True):
    """Render chat messages in specified order"""
    order = "ASC" if chronological else "DESC"
    msgs = enhanced_db.fetch_dicts(conn, f"""
        SELECT role, content, created_at, is_internal
        FROM app.conversation_history
        WHERE session_id=%s AND is_internal=false
        ORDER BY created_at {order};
    """, (session_id,))
    
    for m in msgs:
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(m["content"])

def fetch_chat_messages(conn, session_id: str):
    return enhanced_db.fetch_dicts(conn, """
        SELECT role, content, created_at, is_internal
        FROM app.conversation_history
        WHERE session_id=%s ORDER BY created_at ASC;
    """, (session_id,))

# ---------- main UI ----------
def main_app():
    user = st.session_state[USER_KEY]
    user_id = user["id"]
    user_name = user["full_name"]

    with enhanced_db.session(user_id) as conn:
        enhanced_db.ensure_patient_exists(conn, user_id)

    with st.sidebar:
        st.markdown(f"**👤 {user_name}**  \n{user.get('email','')}")
        if user.get("guardian_email"): 
            st.caption(f"📧 Guardian: {user['guardian_email']}")
        st.divider()

        # Language Selection
        st.markdown("**🌐 Preferred Language**")
        lang_options = {"English 🇬🇧": "en", "Telugu 🇮🇳": "te"}
        current_lang = st.session_state.get("user_selected_lang", "en")
        current_display = "English 🇬🇧" if current_lang == "en" else "Telugu 🇮🇳"
        
        selected_display = st.selectbox(
            "Choose language:",
            options=list(lang_options.keys()),
            index=0 if current_lang == "en" else 1,
            label_visibility="collapsed"
        )
        st.session_state["user_selected_lang"] = lang_options[selected_display]
        
        st.divider()

        session_id = st.session_state.get("consultation_id")
        if session_id:
            if st.button("🔄 Start New Consultation", use_container_width=True, type="secondary"):
                # Finalize current consultation
                with enhanced_db.session(user_id) as conn:
                    enhanced_db.finalize_consultation(conn, session_id)
                
                # Clear session state
                st.session_state.pop("consultation_id", None)
                st.session_state.pop("assistant", None)
                st.session_state.pop("assistant_lang", None)
                st.session_state.pop("consultation_lang", None)
                st.session_state.pop("last_audio_id", None)
                st.session_state.pop("pending_transcription", None)
                st.rerun()
        
        signout_button()

    tab_chat, tab_history = st.tabs(["💬 Consultation", "📋 History"])

    with tab_chat:
        st.header("Health Consultation")
        
        # Show selected language
        current_lang = st.session_state.get("user_selected_lang", "en")
        lang_display = "English 🇬🇧" if current_lang == "en" else "Telugu (తెలుగు) 🇮🇳"
        st.caption(f"Chat in {lang_display} – via text or voice")

        with enhanced_db.session(user_id) as conn:
            active = enhanced_db.get_active_consultation(conn, user_id)

        if active:
            lang_indicator = "🇮🇳 తెలుగు" if get_consultation_language(user_id) == "te" else "🇬🇧 English"
            st.info(f"📌 Active: {active['initial_complaint'][:50]}... | {lang_indicator}")

        st.divider()

        # VOICE INPUT with Confirmation
        if HAS_MIC:
            st.markdown("**🎤 Voice Input**")
            st.caption("Record your message, review transcription, then send")
            
            mic_key = f"mic_{st.session_state.get('consultation_id', 'default')}"
            
            audio = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording", 
                key=mic_key,
                use_container_width=True
            )
            
            # Process audio and show transcription (NO auto-send)
            if audio and audio.get("bytes"):
                import hashlib
                audio_id = hashlib.md5(audio["bytes"]).hexdigest()
                
                # Only transcribe if new audio
                if st.session_state.get("last_audio_id") != audio_id:
                    st.session_state["last_audio_id"] = audio_id
                    
                    with st.spinner("🎧 Transcribing your voice..."):
                        text = transcribe_audio(audio["bytes"], "auto")
                    
                    if text:
                        # Store transcription for review
                        st.session_state["pending_transcription"] = text
                    else:
                        st.warning("⚠️ No speech detected. Please try again.")
                        st.session_state.pop("pending_transcription", None)
            
            # Show transcription and confirmation buttons
            if "pending_transcription" in st.session_state:
                transcribed_text = st.session_state["pending_transcription"]
                detected_lang = detect_lang(transcribed_text)
                lang_emoji = "🇮🇳" if detected_lang == "te" else "🇬🇧"
                lang_name = "Telugu" if detected_lang == "te" else "English"
                
                # Show what was detected
                st.success(f"✅ Transcribed ({lang_emoji} {lang_name}):")
                st.info(f"**{transcribed_text}**")
                
                # Confirmation buttons
                col_send, col_retry = st.columns([1, 1])
                with col_send:
                    if st.button("✅ Send This Message", type="primary", use_container_width=True):
                        st.session_state["pending_voice_message"] = transcribed_text
                        st.session_state.pop("pending_transcription", None)
                        st.rerun()
                
                with col_retry:
                    if st.button("🔄 Record Again", use_container_width=True):
                        st.session_state.pop("pending_transcription", None)
                        st.session_state.pop("last_audio_id", None)
                        st.rerun()
            
            st.divider()
        else:
            st.warning("📦 Install: `pip install streamlit-mic-recorder`")

        # Display conversation (CHRONOLOGICALLY - oldest first, newest at bottom)
        session_id = st.session_state.get("consultation_id")
        if session_id:
            with enhanced_db.session(user_id) as conn:
                msgs = enhanced_db.fetch_dicts(conn, """
                    SELECT role, content, created_at
                    FROM app.conversation_history
                    WHERE session_id=%s AND is_internal=false
                    ORDER BY created_at ASC;
                """, (session_id,))
            
            for m in msgs:
                with st.chat_message("user" if m["role"]=="user" else "assistant"):
                    st.markdown(m["content"])

    with tab_history:
        st.header("Consultation History")
        with enhanced_db.session(user_id) as conn:
            consultations = enhanced_db.fetch_dicts(conn, """
                SELECT id, initial_complaint, status, created_at, completed_at
                FROM app.consultation_sessions
                WHERE patient_id=%s ORDER BY created_at DESC LIMIT 50;
            """, (user_id,))
        
        if not consultations:
            st.info("No history yet. Start your first consultation!")
        else:
            ids = [c["id"] for c in consultations]
            labels = [
                f"{'✅' if c['status']=='completed' else '🔵' if c['status']=='active' else '⛔'} "
                f"{c['initial_complaint'][:60]} – {ist_fmt(c['created_at'])}" 
                for c in consultations
            ]
            sel = st.selectbox("Select:", 
                             options=list(range(len(ids))),
                             format_func=lambda i: labels[i])
            sel_id = ids[sel]

            with enhanced_db.session(user_id) as conn:
                lang_for_sel = get_session_lang(conn, sel_id)

            colA, colB = st.columns([1,1])
            with colA: 
                chronological = st.toggle("Show oldest first (chronological)", value=True)
            with colB: 
                st.caption(f"**{'Telugu' if lang_for_sel=='te' else 'English'}**")

            st.divider()
            with enhanced_db.session(user_id) as conn:
                render_chat_messages(conn, sel_id, chronological=chronological)

            st.divider()
            if st.button("📧 Email Summary"):
                with enhanced_db.session(user_id) as conn:
                    messages = fetch_chat_messages(conn, sel_id)
                
                assistant_hist = get_assistant(lang_for_sel)
                summary_text = assistant_hist.summarize_session(messages)
                
                email_service = get_email_service()
                if email_service.is_enabled():
                    success, msg, recipients = email_service.send_final_recommendation(
                        patient_name=user["full_name"], 
                        patient_email=user["email"],
                        guardian_email=user.get("guardian_email"),
                        recommendation_text=summary_text, 
                        consultation_date=now_ist(), 
                        lang=lang_for_sel,
                    )
                    if success:
                        st.success(f"✅ Sent to: {', '.join(recipients)}")
                    else:
                        st.error(f"❌ {msg}")
                else:
                    st.error("❌ Email not configured")

# ---------- chat input ----------
def handle_chat_input():
    user = st.session_state.get(USER_KEY)
    if not user: return
    user_id = user["id"]

    voice_message = st.session_state.pop("pending_voice_message", None)
    typed_message = st.chat_input("Type your message here…")
    user_input = voice_message or typed_message

    if not user_input: 
        return
    
    user_input = user_input.strip()
    if not user_input: 
        return

    # Use user-selected language or detect from input
    lang = st.session_state.get("user_selected_lang") or detect_lang(user_input)
    
    if "consultation_lang" not in st.session_state:
        st.session_state["consultation_lang"] = lang

    # Thinking animation
    with st.spinner(""):
        timer = st.empty()
        thinking_text = "🤔 ఆలోచిస్తున్నాను…" if lang == "te" else "🤔 Thinking…"
        for n in [3, 2, 1]:
            timer.info(f"{thinking_text} {n}")
            time.sleep(0.3)
        timer.empty()

    with enhanced_db.session(user_id) as conn:
        active = enhanced_db.get_active_consultation(conn, user_id)
        if not active:
            active = enhanced_db.create_consultation_session(conn, user_id, user_input)
            st.session_state["consultation_lang"] = lang
        st.session_state["consultation_id"] = active["id"]

    assistant = get_assistant(lang)
    reply = assistant.process_message(user_input)

    def _to_text(x):
        if isinstance(x, str): return x
        try: return json.dumps(x, ensure_ascii=False)
        except Exception: return str(x)

    with enhanced_db.session(user_id) as conn:
        enhanced_db.save_message(conn, st.session_state["consultation_id"], "user", _to_text(user_input))
        enhanced_db.save_message(conn, st.session_state["consultation_id"], "assistant", _to_text(reply))

    # Check if reply contains recommendations (auto-send email)
    if ("🥼" in reply or "Health Recommendations" in reply or "ఆరోగ్య సిఫార్సులు" in reply) and \
       ("🍽️" in reply or "Diet Plan" in reply or "ఆహార ప్రణాళిక" in reply):
        # This is a final recommendation with diet plan - send email automatically
        success, msg = send_consultation_email(user_id, reply)
        
        # Store email status in session to show notification after rerun
        if success:
            st.session_state["email_sent_notification"] = f"✅ Recommendations sent via email: {msg}"
        else:
            st.session_state["email_sent_notification"] = f"⚠️ Recommendations generated but email failed: {msg}"

    st.rerun()

def run():
    if not auth_gate(): 
        return
    
    # Show email notification if exists
    if "email_sent_notification" in st.session_state:
        notification = st.session_state.pop("email_sent_notification")
        if "✅" in notification:
            st.success(notification)
        else:
            st.warning(notification)
    
    main_app()
    handle_chat_input()

if __name__ == "__main__":
    run()