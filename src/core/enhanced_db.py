# src/core/enhanced_db.py — COMPLETE (JSON-safe saves, sessions/cookies, history)
from __future__ import annotations

import os, hashlib, secrets, json
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional, List, Dict

import psycopg
from psycopg.rows import tuple_row, dict_row
import bcrypt

DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql://neondb_owner:npg_I5jiGFXnWCa1@"
    "ep-calm-dawn-aha4pjmi-pooler.c-3.us-east-1.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

APP_SCHEMA = os.getenv("APP_SCHEMA", "app")
_BOOTSTRAPPED = False
_PASSWORD_COLUMN: Optional[str] = None

@contextmanager
def session(user_id: Optional[str]):
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    try:
        _bootstrap_if_needed(conn)
        if user_id is not None:
            with conn.cursor() as cur:
                cur.execute("select set_config('app.user_id', (%s)::text, true);", (str(user_id),))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def exec_sql(conn, sql: str, params: Optional[Iterable[Any] | Mapping[str, Any]] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)

def fetch_one(conn, sql: str, params: Optional[Iterable[Any] | Mapping[str, Any]] = None):
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(sql, params)
        return cur.fetchone()

def fetch_dicts(conn, sql: str, params: Optional[Iterable[Any] | Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        return list(rows) if rows else []

def _get_password_column(conn) -> str:
    global _PASSWORD_COLUMN
    if _PASSWORD_COLUMN:
        return _PASSWORD_COLUMN
    row = fetch_one(conn, """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name='users'
          AND column_name IN ('pwd_hash','password_hash')
        LIMIT 1;
    """, (APP_SCHEMA,))
    _PASSWORD_COLUMN = row[0] if row else "pwd_hash"
    return _PASSWORD_COLUMN

def _pw_hash(password: str) -> str:
    pre = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.hashpw(pre, bcrypt.gensalt(rounds=12)).decode("utf-8")

def _check_password(password: str, stored_hash: Optional[str]) -> bool:
    if not stored_hash: return False
    try:
        pre = hashlib.sha256(password.encode("utf-8")).digest()
        return bcrypt.checkpw(pre, stored_hash.encode("utf-8"))
    except Exception:
        return False

def _bootstrap_if_needed(conn) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {APP_SCHEMA};")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # users
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.users (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              email citext UNIQUE NOT NULL,
              full_name text,
              created_at timestamptz NOT NULL DEFAULT now()
            );
        """)
        for coldef in ("pwd_hash text", "guardian_email citext", "guardian_name text",
                       "phone text", "age integer", "sex text"):
            try: cur.execute(f"ALTER TABLE {APP_SCHEMA}.users ADD COLUMN IF NOT EXISTS {coldef};")
            except Exception: pass

        # patients
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.patients (
              id uuid PRIMARY KEY,
              full_name text NOT NULL,
              email citext,
              created_at timestamptz NOT NULL DEFAULT now(),
              age integer,
              sex text,
              phone text,
              external_ref uuid
            );
        """)

        # consultations
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.consultation_sessions (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              patient_id uuid NOT NULL REFERENCES {APP_SCHEMA}.patients(id) ON DELETE CASCADE,
              initial_complaint text NOT NULL,
              status text NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','completed','cancelled','ended_by_user')),
              final_diagnosis text,
              final_recommendations text,
              diet_plan_id uuid,
              created_at timestamptz NOT NULL DEFAULT now(),
              completed_at timestamptz
            );
        """)

        # conversation
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.conversation_history (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              session_id uuid NOT NULL REFERENCES {APP_SCHEMA}.consultation_sessions(id) ON DELETE CASCADE,
              role text NOT NULL CHECK (role IN ('user','assistant','system')),
              content text NOT NULL,
              is_internal boolean NOT NULL DEFAULT false,
              created_at timestamptz NOT NULL DEFAULT now()
            );
        """)

        # diet plans
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.diet_plans (
              id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              patient_id uuid NOT NULL REFERENCES {APP_SCHEMA}.patients(id) ON DELETE CASCADE,
              session_id uuid NOT NULL REFERENCES {APP_SCHEMA}.consultation_sessions(id) ON DELETE CASCADE,
              health_condition text,
              inputs jsonb,
              targets jsonb,
              day_plan jsonb,
              rationale text,
              created_at timestamptz NOT NULL DEFAULT now()
            );
        """)

        # email audit
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.email_audit (
              id bigserial PRIMARY KEY,
              patient_id uuid,
              session_id uuid,
              email_type text,
              recipients text,
              subject text,
              status text,
              created_at timestamptz NOT NULL DEFAULT now()
            );
        """)

        # cookie sessions
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.auth_sessions (
              token text PRIMARY KEY,
              user_id uuid NOT NULL REFERENCES {APP_SCHEMA}.users(id) ON DELETE CASCADE,
              created_at timestamptz NOT NULL DEFAULT now(),
              expires_at timestamptz NOT NULL
            );
        """)
    _BOOTSTRAPPED = True

# Users & auth
def create_user(conn, email: str, full_name: str, password: str,
                guardian_email: Optional[str] = None, guardian_name: Optional[str] = None,
                phone: Optional[str] = None, age: Optional[int] = None, sex: Optional[str] = None):
    pw_col = _get_password_column(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            INSERT INTO {APP_SCHEMA}.users (email, full_name, {pw_col}, guardian_email, guardian_name, phone, age, sex)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (email) DO NOTHING
            RETURNING id, email, full_name, guardian_email, guardian_name, phone, age, sex;
        """, (email, full_name, _pw_hash(password), guardian_email, guardian_name, phone, age, sex))
        row = cur.fetchone()
        return dict(row) if row else None

def authenticate_user(conn, email: str, password: str):
    pw_col = _get_password_column(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            SELECT id, email, full_name, {pw_col} AS pwd, guardian_email, guardian_name, phone, age, sex
            FROM {APP_SCHEMA}.users WHERE email=%s;
        """, (email,))
        row = cur.fetchone()
    if not row: return None
    if not _check_password(password, row.get("pwd")): return None
    row.pop("pwd", None)
    return dict(row)

def ensure_patient_exists(conn, user_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH u AS (
                SELECT
                    id,
                    COALESCE(NULLIF(full_name,''), email, 'Unknown') AS full_name,
                    email,
                    age,
                    sex,
                    phone
                FROM {APP_SCHEMA}.users
                WHERE id=%s
            )
            INSERT INTO {APP_SCHEMA}.patients (id, full_name, email, age, sex, phone)
            SELECT id, full_name, email, age, sex, phone FROM u
            ON CONFLICT (id) DO UPDATE SET
                full_name = EXCLUDED.full_name,
                email     = EXCLUDED.email,
                age       = EXCLUDED.age,
                sex       = EXCLUDED.sex,
                phone     = EXCLUDED.phone;
        """, (user_id,))

# Consultations & messages
def create_consultation_session(conn, user_id: str, initial_complaint: str):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            INSERT INTO {APP_SCHEMA}.consultation_sessions (patient_id, initial_complaint)
            VALUES (%s, %s)
            RETURNING *;
        """, (user_id, initial_complaint))
        return dict(cur.fetchone())

def get_active_consultation(conn, user_id: str):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            SELECT * FROM {APP_SCHEMA}.consultation_sessions
            WHERE patient_id=%s AND status='active'
            ORDER BY created_at DESC LIMIT 1;
        """, (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def finalize_consultation(conn, session_id: str):
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {APP_SCHEMA}.consultation_sessions
            SET status='completed', completed_at=NOW()
            WHERE id=%s AND status='active';
        """, (session_id,))

def save_message(conn, session_id: str, role: str, content, internal: bool = False):
    # JSON-safe normalization
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {APP_SCHEMA}.conversation_history (session_id, role, content, is_internal)
            VALUES (%s, %s, %s, %s);
        """, (session_id, role, content, internal))

def get_conversation_history(conn, session_id: str, include_internal: bool = False):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            SELECT role, content, created_at, is_internal
            FROM {APP_SCHEMA}.conversation_history
            WHERE session_id=%s AND (%s OR is_internal=false)
            ORDER BY created_at ASC;
        """, (session_id, include_internal))
        return list(cur.fetchall() or [])

# Email audit
def log_email_to_db(conn, patient_id, session_id, email_type, recipients, subject, status):
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {APP_SCHEMA}.email_audit (patient_id, session_id, email_type, recipients, subject, status)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (patient_id, session_id, email_type, ", ".join(recipients), subject, status))
    except Exception:
        pass

# Cookie sessions
def _ensure_auth_sessions_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.auth_sessions (
              token text PRIMARY KEY,
              user_id uuid NOT NULL REFERENCES {APP_SCHEMA}.users(id) ON DELETE CASCADE,
              created_at timestamptz NOT NULL DEFAULT now(),
              expires_at timestamptz NOT NULL
            );
        """)

def create_auth_session(conn, user_id: str, days: int = 7) -> str:
    _ensure_auth_sessions_table(conn)
    token = secrets.token_urlsafe(32)
    exec_sql(conn, f"""
        INSERT INTO {APP_SCHEMA}.auth_sessions (token, user_id, expires_at)
        VALUES (%s, %s, NOW() + INTERVAL '{days} days');
    """, (token, user_id))
    return token

def get_user_by_session_token(conn, token: str):
    _ensure_auth_sessions_table(conn)
    rows = fetch_dicts(conn, f"""
        SELECT u.id, u.email, u.full_name, u.guardian_email, u.guardian_name, u.phone, u.age, u.sex
        FROM {APP_SCHEMA}.auth_sessions s
        JOIN {APP_SCHEMA}.users u ON u.id = s.user_id
        WHERE s.token=%s AND s.expires_at > NOW()
        LIMIT 1;
    """, (token,))
    return rows[0] if rows else None

def revoke_session(conn, token: str):
    _ensure_auth_sessions_table(conn)
    exec_sql(conn, f"DELETE FROM {APP_SCHEMA}.auth_sessions WHERE token=%s;", (token,))

__all__ = [
    "session", "exec_sql", "fetch_one", "fetch_dicts",
    "create_user", "authenticate_user", "ensure_patient_exists",
    "create_consultation_session", "get_active_consultation",
    "finalize_consultation", "save_message", "get_conversation_history",
    "log_email_to_db",
    "create_auth_session", "get_user_by_session_token", "revoke_session",
]
