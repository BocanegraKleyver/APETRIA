"""Autenticación de un administrador local por instalación.

Almacenamiento SQLite privado en instance/ (nunca versionar).
PBKDF2-HMAC-SHA256 utiliza solo biblioteca estándar y salt aleatorio.
Los tokens de sesión se guardan como hashes SHA-256 (no en texto plano).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, WebSocket

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "instance" / "auth.db"
COOKIE_NAME = "apetria_session"
SESSION_SECONDS = 8 * 60 * 60
PBKDF2_ITERATIONS = 600_000
FAILED_LIMIT = 5
LOCK_SECONDS = 60


@contextmanager
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS administrator (
        id INTEGER PRIMARY KEY CHECK (id=1),
        username TEXT NOT NULL UNIQUE,
        salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        iterations INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        csrf_token TEXT NOT NULL,
        expires_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS login_limits (
        id INTEGER PRIMARY KEY CHECK(id=1),
        attempts INTEGER NOT NULL,
        locked_until INTEGER NOT NULL
    );
    """)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _derive(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()


def _validate_password(password: str) -> None:
    if not 12 <= len(password) <= 256:
        raise ValueError("La contraseña debe tener entre 12 y 256 caracteres")


def has_admin() -> bool:
    with _db() as conn:
        return conn.execute("SELECT 1 FROM administrator WHERE id=1").fetchone() is not None


def create_admin(username: str, password: str) -> None:
    username = username.strip()
    if not (3 <= len(username) <= 48) or not all(c.isascii() and (c.isalnum() or c in "_-.") for c in username):
        raise ValueError("Usuario: 3 a 48 caracteres ASCII (letras, números, guion, punto)")
    _validate_password(password)
    salt = secrets.token_bytes(24)
    with _db() as conn:
        if conn.execute("SELECT 1 FROM administrator WHERE id=1").fetchone():
            raise ValueError("Ya existe un administrador; no se permite crear otro")
        conn.execute(
            "INSERT INTO administrator (id, username, salt, password_hash, iterations) VALUES (1, ?, ?, ?, ?)",
            (username, salt.hex(), _derive(password, salt), PBKDF2_ITERATIONS),
        )


def _matches(row: sqlite3.Row | None, password: str) -> bool:
    if row is None:
        # Reduce las diferencias temporales de una cuenta inexistente.
        _derive(password, b"\x00" * 24)
        return False
    attempt = _derive(password, bytes.fromhex(row["salt"]), row["iterations"])
    return hmac.compare_digest(attempt, row["password_hash"])


def authenticate(username: str, password: str) -> tuple[str, str]:
    now = int(time.time())
    with _db() as conn:
        limit = conn.execute("SELECT attempts, locked_until FROM login_limits WHERE id=1").fetchone()
        if limit and limit["locked_until"] > now:
            raise HTTPException(status_code=429, detail="Demasiados intentos. Intentá nuevamente más tarde")
        row = conn.execute("SELECT * FROM administrator WHERE id=1").fetchone()
        valid = bool(row and hmac.compare_digest(username, row["username"]) and _matches(row, password))
        if not valid:
            if row is None or username != row["username"]:
                _matches(None, password)
            attempts = (limit["attempts"] if limit and limit["locked_until"] <= now else 0) + 1
            locked = now + LOCK_SECONDS if attempts >= FAILED_LIMIT else 0
            conn.execute("INSERT INTO login_limits (id,attempts,locked_until) VALUES(1,?,?) "
                         "ON CONFLICT(id) DO UPDATE SET attempts=excluded.attempts,locked_until=excluded.locked_until",
                         (attempts, locked))
            conn.commit()  # La cuenta de intentos persiste aunque lancemos HTTPException.
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        conn.execute("DELETE FROM login_limits WHERE id=1")
        token = secrets.token_urlsafe(48)
        csrf = secrets.token_urlsafe(32)
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        conn.execute("INSERT INTO sessions(token_hash,csrf_token,expires_at) VALUES(?,?,?)",
                     (_hash_token(token), csrf, now + SESSION_SECONDS))
        return token, csrf


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_for(token: str | None) -> dict[str, Any] | None:
    if not token or len(token) > 512:
        return None
    now = int(time.time())
    with _db() as conn:
        row = conn.execute(
            "SELECT a.username, s.csrf_token, s.expires_at "
            "FROM sessions s JOIN administrator a ON a.id=1 "
            "WHERE s.token_hash=? AND s.expires_at>?",
            (_hash_token(token), now),
        ).fetchone()
        return dict(row) if row else None


def current_session(request: Request) -> dict[str, Any] | None:
    return session_for(request.cookies.get(COOKIE_NAME))


def require_user(request: Request) -> str:
    session = current_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Iniciá sesión para continuar")
    return session["username"]


def require_csrf(request: Request) -> str:
    username = require_user(request)
    session = current_session(request)
    assert session is not None
    supplied = request.headers.get("x-csrf-token", "")
    if not supplied or not hmac.compare_digest(supplied, session["csrf_token"]):
        raise HTTPException(status_code=403, detail="Token CSRF ausente o inválido")
    return username


def ws_allowed(websocket: WebSocket) -> bool:
    # Los navegadores envían Origin. Rechazar un origen distinto impide
    # conexiones WebSocket iniciadas desde otros sitios, incluso si portan
    # accidentalmente la cookie del administrador.
    origin = websocket.headers.get("origin")
    if origin:
        scheme = "https" if websocket.url.scheme == "wss" else "http"
        expected = f"{scheme}://{websocket.headers.get('host', '')}"
        if origin != expected:
            return False
    # Los clientes de prueba/CLI sin Origin pueden seguir autenticándose
    # mediante la cookie de sesión, como PF-13.
    return session_for(websocket.cookies.get(COOKIE_NAME)) is not None


def revoke(token: str | None) -> None:
    if not token:
        return
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (_hash_token(token),))


def change_password(current: str, new_password: str) -> None:
    _validate_password(new_password)
    with _db() as conn:
        row = conn.execute("SELECT * FROM administrator WHERE id=1").fetchone()
        if not _matches(row, current):
            raise HTTPException(status_code=401, detail="Contraseña actual incorrecta")
        salt = secrets.token_bytes(24)
        conn.execute("UPDATE administrator SET salt=?, password_hash=?, iterations=? WHERE id=1",
                     (salt.hex(), _derive(new_password, salt), PBKDF2_ITERATIONS))
        conn.execute("DELETE FROM sessions")
