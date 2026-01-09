import sqlite3
import os
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "interview.db")

def init_db():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # UPDATED: Creates table with all 9 columns needed for the full project
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            question TEXT,
            answer_text TEXT,
            tech_score INTEGER,
            feedback TEXT,
            posture TEXT,
            light TEXT
        )
    ''')
    conn.commit()
    conn.close()

# FIX: Now accepts all 7 arguments passed by app.py
def save_session(role, question, answer, score, feedback, posture, light):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO results 
        (timestamp, role, question, answer_text, tech_score, feedback, posture, light)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, role, question, answer, score, feedback, posture, light))
    
    conn.commit()
    conn.close()

def get_all_sessions():
    if not os.path.exists(DB_PATH): return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # We select only the columns the Sidebar expects to avoid dataframe errors
    c.execute("SELECT id, timestamp, role, question, answer_text, tech_score FROM results ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data

init_db()