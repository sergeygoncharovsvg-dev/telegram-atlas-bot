import os
import re
import sqlite3
import time
from typing import List, Dict, Tuple, Optional

import requests
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# =========================
# Config (env vars)
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# "OpenAI-compatible" chat endpoint (many providers support this pattern).
# You can keep it as-is if your provider is compatible; otherwise change it.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()  # e.g. https://api.yourprovider.com/v1/chat/completions
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()  # provider-specific model name
LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "60"))

DB_PATH = os.getenv("DB_PATH", "memory.sqlite")
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "14"))  # stored messages per chat (user+assistant)

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")
if not LLM_BASE_URL:
    raise RuntimeError("Missing LLM_BASE_URL env var.")
if not LLM_API_KEY:
    raise RuntimeError("Missing LLM_API_KEY env var.")


# =========================
# DB helpers
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,            -- 'user' or 'assistant'
                content TEXT NOT NULL,
                ts INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_ts
            ON messages(chat_id, ts);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                ts INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pins_chat_ts
            ON pins(chat_id, ts);
        """)

def add_message(chat_id: int, role: str, content: str) -> None:
    now = int(time.time())
    with db() as conn:
        conn.execute(
            "INSERT INTO messages(chat_id, role, content, ts) VALUES (?,?,?,?)",
            (chat_id, role, content, now),
        )
        # Trim history
        cur = conn.execute(
            "SELECT id FROM messages WHERE chat_id=? ORDER BY ts DESC, id DESC LIMIT ?",
            (chat_id, HISTORY_TURNS),
        )
        keep_ids = {row[0] for row in cur.fetchall()}
        conn.execute(
            "DELETE FROM messages WHERE chat_id=? AND id NOT IN ({})".format(
                ",".join(["?"] * len(keep_ids)) if keep_ids else "NULL"
            ),
            (chat_id, *keep_ids) if keep_ids else (chat_id,),
        )

def get_history(chat_id: int) -> List[Dict[str, str]]:
    with db() as conn:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY ts ASC, id ASC",
            (chat_id,),
        )
        return [{"role": r, "content": c} for (r, c) in cur.fetchall()]

def clear_history(chat_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))

def add_pin(chat_id: int, content: str) -> None:
    now = int(time.time())
    with db() as conn:
        conn.execute(
            "INSERT INTO pins(chat_id, content, ts) VALUES (?,?,?)",
            (chat_id, content, now),
        )

def recall_pins(chat_id: int, query: str, limit: int = 10) -> List[Tuple[int, str]]:
    q = f"%{query.strip()}%"
    with db() as conn:
        cur = conn.execute(
            "SELECT id, content FROM pins WHERE chat_id=? AND content LIKE ? ORDER BY ts DESC LIMIT ?",
            (chat_id, q, limit),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

def get_recent_pins(chat_id: int, limit: int = 5) -> List[str]:
    with db() as conn:
        cur = conn.execute(
            "SELECT content FROM pins WHERE chat_id=? ORDER BY ts DESC LIMIT ?",
            (chat_id, limit),
        )
        return [row[0] for row in cur.fetchall()]


# =========================
# LLM call (OpenAI-compatible)
# =========================
SYSTEM_PROMPT = """You are Atlas, the user's strategic AI assistant.
Rules:
- Reply ALWAYS in two sections: (1) English, then (2) Russian.
- Keep answers practical, structured, and concise.
- If the user asks for a plan, give steps + checklist.
- If you are uncertain, say what you assume.
- Do NOT claim you have real feelings or consciousness.
"""

def build_messages(chat_id: int, user_text: str) -> List[Dict[str, str]]:
    history = get_history(chat_id)
    pins = get_recent_pins(chat_id, limit=5)

    context_block = ""
    if pins:
        context_block = "Pinned notes (high priority context):\n- " + "\n- ".join(pins)

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_block:
        msgs.append({"role": "system", "content": context_block})

    # Append chat history
    msgs.extend(history)

    # New user message
    msgs.append({"role": "user", "content": user_text})
    return msgs

def call_llm(messages: List[Dict[str, str]]) -> str:
    # OpenAI-compatible Chat Completions payload
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.4,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(LLM_BASE_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()

    # Typical shape: { choices: [ { message: { content: "..." } } ] }
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback: show raw if provider differs
        return str(data)


# =========================
# Telegram handlers
# =========================
HELP_TEXT = (
    "Commands:\n"
    "/new — reset this chat's memory\n"
    "/pin <text> — save a note (e.g., preferences, projects)\n"
    "/recall <word> — search your saved notes\n"
    "/help — show this\n\n"
    "Just send a message and I’ll reply in English + Russian."
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Hi — I’m Atlas in Telegram.\n\n" + HELP_TEXT)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT)

async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.message.reply_text("✅ Reset done. (English + Russian replies will continue.)")

async def cmd_pin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    m = re.match(r"^/pin\s+(.+)$", text, flags=re.DOTALL)
    if not m:
        await update.message.reply_text("Usage: /pin <text to remember>")
        return
    note = m.group(1).strip()
    if len(note) < 2:
        await update.message.reply_text("Note is too short. Try again.")
        return
    add_pin(chat_id, note)
    await update.message.reply_text("📌 Saved.")

async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    m = re.match(r"^/recall\s+(.+)$", text, flags=re.DOTALL)
    if not m:
        await update.message.reply_text("Usage: /recall <keyword>")
        return
    q = m.group(1).strip()
    results = recall_pins(chat_id, q, limit=10)
    if not results:
        await update.message.reply_text("No matches found.")
        return
    lines = [f"- {content}" for (_id, content) in results]
    await update.message.reply_text("Matches:\n" + "\n".join(lines))

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    # Show "typing..."
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Store user msg
    add_message(chat_id, "user", user_text)

    # Build + call LLM
    try:
        msgs = build_messages(chat_id, user_text)
        assistant_text = call_llm(msgs)
    except Exception as e:
        assistant_text = (
            "English:\n"
            "I hit an error calling the AI API. Check LLM_BASE_URL / LLM_API_KEY / model.\n"
            f"Error: {e}\n\n"
            "Russian:\n"
            "Произошла ошибка при вызове AI API. Проверьте LLM_BASE_URL / LLM_API_KEY / модель.\n"
            f"Ошибка: {e}"
        )

    # Store assistant msg
    add_message(chat_id, "assistant", assistant_text)

    # Reply
    await update.message.reply_text(assistant_text)


def main() -> None:
    init_db()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("pin", cmd_pin))
    app.add_handler(CommandHandler("recall", cmd_recall))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    # Polling is simplest for MVP
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
