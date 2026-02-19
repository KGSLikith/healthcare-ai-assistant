# app/enhanced_app.py
from __future__ import annotations

import os, sys
from datetime import datetime as dt
from zoneinfo import ZoneInfo
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Optional mic recorder
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# Path bootstrap
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our enhanced modules (you'll need to adjust these paths)
# For now, using the enhanced versions we created
import sys
sys.path.insert(0, "/home/claude")

from enhanced_db import (
    session, fetch_dicts, exec_sql, fetch_one,
    create_user, authenticate_user, get_user_details,
    ensure_patient_exists, create_consultation_session,
    get_active_consultation, finalize_consultation,
    save_message, get_conversation_history
)
from intelligent_assistant import HealthAssistant
from email_service import get_email_service
from psycopg.types.json import Json

# ==============================================================================
# Config
# ==============================================================================
load_dotenv()
st.set_page_config(page_title="Health AI Assistant", page_icon="🏥", layout="wide")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

USER_KEY = "auth_user"
IST = ZoneInfo("Asia/Kolkata")

def now_ist() -> dt:
    return dt.now(IST)

def ist_fmt(d: dt) -> str:
    return d.astimezone(IST).strftime("%a, %d %b %Y %I:%M %p IST")

# Language detection
def detect_lang(text: str) -> str:
    if any('\u0c00' <= ch <= '\u0c7f' for ch in text or ""):
        return "te"
    return "en"

# ==============================================================================
# Authentication
# ==============================================================================
def auth_gate():
    st.title("🏥 Health AI Assistant")
    st.caption("Your intelligent health companion - Available in English & Telugu")
    
    tab_login, tab_signup = st.tabs(["🔑 Sign In", "✨ Create Account"])
    
    with tab_login:
        st.subheader("Welcome Back")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
        
        if submit:
            if not email or not pwd:
                st.error("Please enter both email and password")
            else:
                try:
                    with session(None) as conn:
                        user = authenticate_user(conn, email, pwd)
                    
                    if not user:
                        st.error("❌ Invalid email or password")
                    else:
                        st.session_state[USER_KEY] = user
                        st.success(f"✅ Welcome back, {user['full_name']}!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Login error: {e}")
    
    with tab_signup:
        st.subheader("Create Your Account")
        st.caption("We'll keep your health data private and secure")
        
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input("Full Name *", placeholder="John Doe")
                email = st.text_input("Email *", placeholder="your.email@example.com")
                pwd = st.text_input("Password *", type="password", 
                                   help="Minimum 8 characters")
            
            with col2:
                guardian_name = st.text_input("Guardian Name (Optional)", 
                                             placeholder="Family member or caregiver")
                guardian_email = st.text_input("Guardian Email (Optional)",
                                              placeholder="guardian@example.com",
                                              help="We'll send important alerts to this email too")
                phone = st.text_input("Phone (Optional)", placeholder="+91 98765 43210")
            
            col3, col4 = st.columns(2)
            with col3:
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
            with col4:
                sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            
            st.caption("* Required fields")
            submit = st.form_submit_button("Create Account", use_container_width=True)
        
        if submit:
            if not all([full_name, email, pwd]):
                st.error("Please fill all required fields (*)")
            elif len(pwd) < 8:
                st.error("Password must be at least 8 characters")
            else:
                try:
                    with session(None) as conn:
                        user = create_user(
                            conn, email, full_name, pwd,
                            guardian_email=guardian_email or None,
                            guardian_name=guardian_name or None,
                            phone=phone or None,
                            age=age,
                            sex=sex.lower()
                        )
                    
                    if not user:
                        st.error("❌ This email is already registered")
                    else:
                        st.session_state[USER_KEY] = user
                        st.success("✅ Account created successfully!")
                        if guardian_email:
                            st.info(f"📧 Guardian alerts will be sent to: {guardian_email}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Registration error: {e}")

# ==============================================================================
# Main Application
# ==============================================================================
def main_app():
    user = st.session_state[USER_KEY]
    user_id = user["id"]
    user_name = user["full_name"]
    
    # Ensure patient exists
    with session(user_id) as conn:
        ensure_patient_exists(conn, user_id)
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Health AI")
        st.caption(f"👤 {user_name}")
        
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # User info
        st.caption("**Account Details**")
        if user.get("age"):
            st.caption(f"Age: {user['age']}")
        if user.get("sex"):
            st.caption(f"Sex: {user['sex'].title()}")
        if user.get("guardian_email"):
            st.caption(f"📧 Guardian: {user['guardian_email']}")
        
        st.divider()
        
        # Quick stats
        with session(user_id) as conn:
            stats = fetch_one(conn, """
                select 
                    (select count(*) from app.consultation_sessions where patient_id=%s) as consultations,
                    (select count(*) from app.diet_plans where patient_id=%s) as diet_plans
            """, (user_id, user_id))
            
            if stats:
                st.metric("Total Consultations", stats[0])
                st.metric("Diet Plans", stats[1])
    
    # Main tabs
    tab_chat, tab_history = st.tabs(["💬 Health Consultation", "📋 History"])
    
    # ========== CHAT TAB ==========
    with tab_chat:
        st.header("Health Consultation")
        st.caption("Describe your symptoms and I'll help guide you. Speak in English or Telugu.")
        
        # Check for active consultation
        with session(user_id) as conn:
            active_consultation = get_active_consultation(conn, user_id)
        
        # New consultation button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🆕 Start New Consultation", disabled=bool(active_consultation)):
                st.session_state.pop("assistant", None)
                st.session_state.pop("consultation_id", None)
                st.rerun()
        
        with col2:
            if active_consultation and st.button("🔄 Continue Active"):
                st.session_state["consultation_id"] = active_consultation["id"]
                st.rerun()
        
        with col3:
            if active_consultation and st.button("✅ Complete Now"):
                st.session_state["force_finalize"] = True
                st.rerun()
        
        if active_consultation:
            st.info(f"📌 Active consultation: {active_consultation['initial_complaint'][:50]}...")
        
        # Voice input section
        if HAS_MIC:
            with st.expander("🎤 Voice Input (Telugu/English)"):
                col_v1, col_v2 = st.columns([3, 1])
                
                with col_v1:
                    audio = mic_recorder(
                        key="mic_input",
                        start_prompt="🎙️ Start Recording",
                        stop_prompt="⏹️ Stop Recording"
                    )
                
                with col_v2:
                    lang_hint = st.selectbox("Language", ["Auto", "Telugu", "English"])
                
                if audio and audio.get("bytes"):
                    st.warning("🔧 Voice transcription requires Whisper API setup")
                    # TODO: Implement actual transcription
                    # For now, users can type
        
        # Display conversation
        st.divider()
        
        consultation_id = st.session_state.get("consultation_id")
        if consultation_id:
            with session(user_id) as conn:
                messages = get_conversation_history(conn, consultation_id, include_internal=False)
            
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(msg["content"])
        
        # Chat input (OUTSIDE any container for Streamlit to work properly)
        pass  # We'll handle this after the tab definition

    # ========== HISTORY TAB ==========
    with tab_history:
        st.header("Consultation History")
        
        with session(user_id) as conn:
            consultations = fetch_dicts(conn, """
                select 
                    id, 
                    initial_complaint,
                    status,
                    created_at,
                    completed_at,
                    final_diagnosis
                from app.consultation_sessions
                where patient_id = %s
                order by created_at desc
                limit 20
            """, (user_id,))
        
        if not consultations:
            st.info("No consultation history yet. Start your first consultation in the chat tab!")
        else:
            for consult in consultations:
                with st.expander(
                    f"{'✅' if consult['status'] == 'completed' else '🔄'} "
                    f"{consult['initial_complaint'][:60]}... - {ist_fmt(consult['created_at'])}"
                ):
                    st.caption(f"**Status:** {consult['status'].title()}")
                    st.caption(f"**Started:** {ist_fmt(consult['created_at'])}")
                    
                    if consult['completed_at']:
                        st.caption(f"**Completed:** {ist_fmt(consult['completed_at'])}")
                    
                    if consult['final_diagnosis']:
                        st.markdown("**Final Assessment:**")
                        st.markdown(consult['final_diagnosis'][:200] + "...")
                    
                    # View full conversation
                    if st.button(f"View Full Conversation", key=f"view_{consult['id']}"):
                        with session(user_id) as conn:
                            msgs = get_conversation_history(conn, consult['id'])
                        
                        for msg in msgs:
                            role_emoji = "👤" if msg["role"] == "user" else "🤖"
                            st.caption(f"{role_emoji} **{msg['role'].title()}:** {msg['content'][:100]}...")

# ==============================================================================
# Chat Input Handler (Must be at root level)
# ==============================================================================
def handle_chat_input():
    """Handle chat input - must be called at root level"""
    user = st.session_state.get(USER_KEY)
    if not user:
        return
    
    user_id = user["id"]
    
    # Check for pending voice input
    user_input = st.session_state.pop("pending_voice_input", None)
    
    # Or get from chat_input
    if not user_input:
        user_input = st.chat_input("Type your message here...", key="main_chat_input")
    
    if not user_input:
        # Check for forced finalization
        if st.session_state.pop("force_finalize", False):
            finalize_current_consultation(user_id)
        return
    
    user_input = user_input.strip()
    if not user_input:
        return
    
    # Detect language
    lang = detect_lang(user_input)
    
    # Get or create consultation
    with session(user_id) as conn:
        consultation = get_active_consultation(conn, user_id)
        
        if not consultation:
            # Create new consultation
            consultation_id = create_consultation_session(conn, user_id, user_input[:100])
            st.session_state["consultation_id"] = consultation_id
        else:
            consultation_id = consultation["id"]
            st.session_state["consultation_id"] = consultation_id
        
        # Save user message
        save_message(conn, consultation_id, "user", user_input)
    
    # Get or create assistant
    assistant: HealthAssistant = st.session_state.get("assistant")
    if not assistant:
        with session(user_id) as conn:
            user_details = get_user_details(conn, user_id)
        
        assistant = HealthAssistant(
            groq_client=groq_client,
            model=GROQ_MODEL,
            lang=lang,
            user_context={
                "age": user_details.get("age"),
                "sex": user_details.get("sex"),
            }
        )
        
        # Load conversation history
        with session(user_id) as conn:
            history = get_conversation_history(conn, consultation_id)
            for msg in history[:-1]:  # Exclude the message we just added
                assistant.conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        st.session_state["assistant"] = assistant
    
    # Process message
    with st.spinner("Thinking..." if lang == "en" else "ఆలోచిస్తున్నాను..."):
        try:
            response = assistant.process_message(user_input)
            
            # Save assistant response
            with session(user_id) as conn:
                save_message(conn, consultation_id, "assistant", response["response"])
            
            # Check if we should attempt finalization
            if response.get("should_check_finalization"):
                finalization = assistant.attempt_finalization()
                
                if finalization.get("ready"):
                    # Finalize consultation
                    finalize_consultation_with_analysis(
                        user_id, 
                        consultation_id, 
                        finalization["analysis"],
                        lang
                    )
                else:
                    # Not ready yet, continue conversation
                    pass
            
        except Exception as e:
            st.error(f"Error processing message: {e}")
            # Save error message
            error_msg = ("క్షమించండి, దోషం సంభవించింది. దయచేసి మళ్లీ ప్రయత్నించండి." 
                        if lang == "te" else 
                        "Sorry, an error occurred. Please try again.")
            with session(user_id) as conn:
                save_message(conn, consultation_id, "assistant", error_msg)
    
    st.rerun()

def finalize_consultation_with_analysis(user_id: str, consultation_id: str, analysis: dict, lang: str):
    """Finalize consultation with LLM analysis"""
    assistant: HealthAssistant = st.session_state.get("assistant")
    
    # Format recommendation
    recommendation_text = assistant.format_final_recommendation(analysis)
    
    # Save final message
    with session(user_id) as conn:
        save_message(conn, consultation_id, "assistant", recommendation_text)
    
    # Generate diet plan if needed
    diet_plan_id = None
    if analysis.get("diet_needed"):
        with session(user_id) as conn:
            user_details = get_user_details(conn, user_id)
            
            user_profile = {
                "age": user_details.get("age", 30),
                "sex": user_details.get("sex", "not specified"),
                "weight": 70,  # TODO: Get from user profile
                "height": 170,  # TODO: Get from user profile
                "activity_level": "moderate"  # TODO: Get from user profile
            }
            
            diet_plan = assistant.generate_diet_plan(
                health_condition=analysis.get("diet_context", "General health"),
                user_profile=user_profile
            )
            
            # Save diet plan
            rows = fetch_dicts(conn, """
                insert into app.diet_plans 
                (patient_id, session_id, health_condition, targets, day_plan, rationale)
                values (%s, %s, %s, %s, %s, %s)
                returning id;
            """, (
                user_id, 
                consultation_id,
                analysis.get("diet_context", "General health"),
                Json(diet_plan.get("targets", {})),
                Json(diet_plan.get("day_plan", {})),
                diet_plan.get("rationale")
            ))
            diet_plan_id = rows[0]["id"] if rows else None
    
    # Finalize in database
    with session(user_id) as conn:
        summary_text = analysis.get("summary", {}).get("chief_complaint", "Health consultation")
        finalize_consultation(
            conn,
            consultation_id,
            final_diagnosis=summary_text,
            recommendations=recommendation_text,
            diet_plan_id=diet_plan_id
        )
        
        # Get user details for email
        user_details = get_user_details(conn, user_id)
    
    # Send emails
    email_service = get_email_service()
    if email_service.is_enabled():
        # Send final recommendation
        success, msg, recipients = email_service.send_final_recommendation(
            patient_name=user_details["full_name"],
            patient_email=user_details["email"],
            guardian_email=user_details.get("guardian_email"),
            recommendation_text=recommendation_text,
            consultation_date=now_ist()
        )
        
        # Log email
        with session(user_id) as conn:
            email_service.log_email_to_db(
                conn, user_id, consultation_id,
                "final_recommendation", recipients, 
                f"Health Consultation Summary",
                "sent" if success else "failed"
            )
        
        # Send diet plan if created
        if diet_plan_id and analysis.get("diet_needed"):
            success, msg, recipients = email_service.send_diet_plan(
                patient_name=user_details["full_name"],
                patient_email=user_details["email"],
                guardian_email=user_details.get("guardian_email"),
                diet_plan=diet_plan,
                health_condition=analysis.get("diet_context", "General health"),
                consultation_date=now_ist()
            )
            
            with session(user_id) as conn:
                email_service.log_email_to_db(
                    conn, user_id, consultation_id,
                    "diet_plan", recipients,
                    f"Personalized Diet Plan",
                    "sent" if success else "failed"
                )
        
        st.success("✅ Consultation completed! Summary and diet plan sent to your email" + 
                  (" and guardian" if user_details.get("guardian_email") else ""))
    else:
        st.warning("⚠️ Email not configured - recommendations saved but not emailed")
    
    # Clear session state
    st.session_state.pop("assistant", None)
    st.session_state.pop("consultation_id", None)

# ==============================================================================
# App Entry Point
# ==============================================================================
if __name__ == "__main__":
    # Check authentication
    user = st.session_state.get(USER_KEY)
    
    if not user:
        auth_gate()
    else:
        main_app()
        
        # Handle chat input at root level (CRITICAL for Streamlit)
        handle_chat_input()