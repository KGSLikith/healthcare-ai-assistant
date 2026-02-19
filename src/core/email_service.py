# src/core/email_service.py — COMPLETE
import os, smtplib, ssl
from email.message import EmailMessage
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class EmailConfig:
    host: str
    port: int
    user: str
    password: str
    from_name: str
    from_email: str
    secure: str  # 'ssl' | 'starttls' | 'none'
    override_recipients: List[str]

    @classmethod
    def from_env(cls) -> "EmailConfig":
        return cls(
            host=os.getenv("SMTP_HOST", "").strip(),
            port=int(os.getenv("SMTP_PORT", "587") or 587),
            user=os.getenv("SMTP_USER", "").strip(),
            password=(os.getenv("SMTP_PASS") or os.getenv("SMTP_PASSWORD") or "").strip(),
            from_name=os.getenv("SMTP_FROM_NAME", "Health AI Assistant").strip(),
            from_email=(os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USER", "")).strip(),
            secure=os.getenv("SMTP_SECURE", "starttls").strip().lower(),
            override_recipients=[e.strip() for e in os.getenv("SMTP_TO_OVERRIDE", "").split(",") if e.strip()],
        )

    def is_configured(self) -> bool:
        return bool(self.host and self.port and self.from_email)

class EmailService:
    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or EmailConfig.from_env()

    def is_enabled(self) -> bool:
        return self.config.is_configured()

    def _send_email(self, subject: str, body: str, recipients: List[str]) -> Tuple[bool, str]:
        if not self.is_enabled():
            return False, "Email not configured"
        to_list = self.config.override_recipients or recipients
        to_list = [r for r in to_list if r and r.strip()]
        if not to_list:
            return False, "No valid recipients"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"{self.config.from_name} <{self.config.from_email}>"
        msg["To"] = ", ".join(to_list)
        msg.set_content(body)

        try:
            if self.config.secure == "ssl":
                with smtplib.SMTP_SSL(self.config.host, self.config.port, context=ssl.create_default_context()) as s:
                    if self.config.user:
                        s.login(self.config.user, self.config.password)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(self.config.host, self.config.port) as s:
                    if self.config.secure == "starttls":
                        s.starttls(context=ssl.create_default_context())
                    if self.config.user:
                        s.login(self.config.user, self.config.password)
                    s.send_message(msg)
            return True, "sent"
        except Exception as e:
            return False, str(e)

    def send_final_recommendation(
        self,
        patient_name: str,
        patient_email: str,
        guardian_email: Optional[str],
        recommendation_text: str,
        consultation_date: datetime,
        lang: str = "en",
    ) -> Tuple[bool, str, List[str]]:
        subject = ("ఆరోగ్య సలహా సారాంశం" if lang == "te" else "Health Consultation Summary") \
                  + f" - {consultation_date.strftime('%B %d, %Y')}"
        if lang == "te":
            body = f"""ప్రియమైన {patient_name},

మీ ఆరోగ్య పరామర్శ సారాంశం:

{recommendation_text}

---
తేదీ: {consultation_date.strftime('%A, %B %d, %Y %I:%M %p IST')}
గమనిక: ఇది వైద్య నిర్ధారణ కాదు; లక్షణాలు తీవ్రమైతే వెంటనే వైద్య సహాయం పొందండి.

– Health AI Assistant
"""
        else:
            body = f"""Dear {patient_name},

Here is the summary of your health consultation:

{recommendation_text}

---
Date: {consultation_date.strftime('%A, %B %d, %Y %I:%M %p IST')}
Note: This is educational guidance, not a diagnosis. Seek care if symptoms are severe.

– Health AI Assistant
"""
        recipients = [patient_email] + ([guardian_email] if guardian_email else [])
        success, message = self._send_email(subject, body, recipients)
        return success, message, recipients

    def log_email_to_db(self, conn, patient_id, session_id, email_type, recipients, subject, status):
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO app.email_audit (patient_id, session_id, email_type, recipients, subject, status)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (patient_id, session_id, email_type, ", ".join(recipients), subject, status))
        except Exception:
            pass

_email_service: Optional[EmailService] = None
def get_email_service() -> EmailService:
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
