# %%
import psycopg2
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import os
from dotenv import load_dotenv
import json

load_dotenv()
load_dotenv("../../.env")  # remove in Docker

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        dbname=os.getenv("POSTGRES_DB").strip('"'),
        user=os.getenv("POSTGRES_USER").strip('"'),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def create_application_logs():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS application_logs (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT PRIMARY KEY,
            current_q_index INTEGER DEFAULT 0
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS retrieved_docs (
            session_id TEXT PRIMARY KEY,
            docs JSONB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    conn.commit()
    cur.close()
    conn.close()

def insert_message(session_id, message, model):
    if message.type not in ("human", "ai"):
        return  # Ignore system/tool messages for now

    # Skip empty AI messages that only contain tool calls
    if (
        message.type == "ai"
        and not getattr(message, "content", "")
        and getattr(message, "tool_calls", None)
    ):
        return

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO application_logs (session_id, role, content, model)
        VALUES (%s, %s, %s, %s)
    ''', (
        session_id,
        message.type,
        message.content,
        model
    ))
    conn.commit()
    cur.close()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT role, content FROM application_logs
        WHERE session_id = %s
        ORDER BY created_at
    ''', (session_id,))
    messages = []
    for role, content in cur.fetchall():
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    cur.close()
    conn.close()
    return messages

def get_current_q_index(session_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO session_state (session_id, current_q_index)
        VALUES (%s, 0)
        ON CONFLICT (session_id) DO NOTHING;
    ''', (session_id,))
    cur.execute('''
        SELECT current_q_index FROM session_state WHERE session_id = %s
    ''', (session_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] 

def increment_q_index(session_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Insert default if missing
    cur.execute('''
        INSERT INTO session_state (session_id, current_q_index)
        VALUES (%s, 0)
        ON CONFLICT (session_id) DO NOTHING;
    ''', (session_id,))

    cur.execute('''
        UPDATE session_state 
        SET current_q_index = current_q_index + 1 
        WHERE session_id = %s
    ''', (session_id,))
    conn.commit()
    cur.close()
    conn.close()

def save_retrieved_docs(session_id: str, docs: list[dict]):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO retrieved_docs (session_id, docs, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (session_id) DO UPDATE SET docs = EXCLUDED.docs, updated_at = CURRENT_TIMESTAMP;
    ''', (session_id, json.dumps(docs)))
    conn.commit()
    cur.close()
    conn.close()

def get_retrieved_docs(session_id: str) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT docs FROM retrieved_docs WHERE session_id = %s
    ''', (session_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row and row[0]:
        return row[0]  # Already deserialized from JSONB
    return []

def clear_chat_history(session_id: str = None):
    """Deletes chat history. Raises error if nothing is stored or session not found."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        if session_id:
            # Check if the session exists
            cur.execute("SELECT 1 FROM application_logs WHERE session_id = %s LIMIT 1;", (session_id,))
            exists = cur.fetchone()
            if not exists:
                raise ValueError(f"(!) No chat history found for session: {session_id}")

            # Proceed to delete
            cur.execute("DELETE FROM application_logs WHERE session_id = %s;", (session_id,))
            cur.execute("DELETE FROM session_state WHERE session_id = %s;", (session_id,))
            cur.execute("DELETE FROM retrieved_docs WHERE session_id = %s;", (session_id,))
            print(f"🧹 Cleared chat history for session: {session_id}")
        else:
            # Check if anything exists at all
            cur.execute("SELECT COUNT(*) FROM application_logs;")
            count = cur.fetchone()[0]
            if count == 0:
                raise ValueError("(!) No chat history found to delete.")

            cur.execute("DELETE FROM application_logs;")
            cur.execute("DELETE FROM session_state;")
            cur.execute("DELETE FROM retrieved_docs;")
            print("🧹 Cleared all chat history.")
        
        conn.commit()

    finally:
        cur.close()
        conn.close()

def check_session_history(session_id: str = None):
    """Prints chat history and current question index.
    If session_id is None, prints data for all sessions.
    """
    def get_all_session_ids() -> list[str]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT DISTINCT session_id FROM application_logs;")
            rows = cur.fetchall()
            return [row[0] for row in rows if row[0] is not None]
        finally:
            cur.close()
            conn.close()

    session_ids = [session_id] if session_id else get_all_session_ids()
    
    if not session_ids:
        print("(!) No session data found.")
        return
    
    for sid in session_ids:
        try:
            messages = get_chat_history(sid)
            print(f"\n# --< Chat history for session {sid} >------------------------------------------------------------:\n")
            for msg in messages:
                role = msg.__class__.__name__.replace("Message", "")
                print(f"[{role}] {msg.content}\n")

            current_index = get_current_q_index(sid)
            print(f"Current Question Index: {current_index}")
        
            # ➕ Show retrieved docs
            docs = get_retrieved_docs(sid)
            if docs:
                print(f"\nRetrieved Docs ({len(docs)} total):")
                for i, doc in enumerate(docs):
                    snippet = doc.get("page_content", "")[:100].replace("\n", " ")
                    score = doc.get("score", "N/A")
                    print(f"  {i+1}. Score: {score:.2f} | {snippet}...")
            else:
                print("\nNo retrieved docs found for this session.")

        except Exception as e:
            print(f"(!) Error processing session {sid}: {e}")


# Init
create_application_logs()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        try:
            conn = get_db_connection()
            print("Successfully connected to the database")
            conn.close()
            print("Usage:")
            print("  python db_utils.py clear [session_id]  # Clear all or specific session history")
            print("  python db_utils.py check [session_id]  # Show all or specific session history")
            
        except Exception as e:
            print(f"Connection error: {e}")
        
        sys.exit(1)

    action = sys.argv[1]

    if action == "clear":
        if len(sys.argv) == 3:
            session_id = sys.argv[2]
            confirm = input(f"❗ Delete chat history for session '{session_id}'? Type 'yes' to confirm: ")
            if confirm.lower() == "yes":
                clear_chat_history(session_id)
            else:
                print("🛑 Deletion cancelled.")
        else:
            confirm = input("❗ Delete ALL chat history? Type 'yes' to confirm: ")
            if confirm.lower() == "yes":
                clear_chat_history()
            else:
                print("🛑 Deletion cancelled.")

    elif action == "check":
        session_id = sys.argv[2] if len(sys.argv) == 3 else None
        check_session_history(session_id)

    else:
        print(f"❌ Unknown command: {action}")




# %%
