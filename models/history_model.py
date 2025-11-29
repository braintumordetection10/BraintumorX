import sqlite3

DB_PATH = "history.db"


# ---------------------------
# Initialize DB + create table
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path    TEXT,
            prediction    TEXT,
            confidence    REAL,
            heatmap_path  TEXT,
            overlay_path  TEXT,
            probabilities TEXT,
            patient_name  TEXT,
            patient_age   INTEGER,
            patient_gender TEXT,
            created_at    TEXT
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------
# Insert new record
# ---------------------------
def add_record(image_path,
               prediction,
               confidence,
               heatmap_path,
               overlay_path,
               probabilities,
               patient_name,
               patient_age,
               patient_gender):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history (
            image_path,
            prediction,
            confidence,
            heatmap_path,
            overlay_path,
            probabilities,
            patient_name,
            patient_age,
            patient_gender,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        image_path,
        prediction,
        confidence,
        heatmap_path,
        overlay_path,
        str(probabilities),
        patient_name,
        patient_age,
        patient_gender
    ))

    conn.commit()
    conn.close()


# ---------------------------
# Retrieve all records
# ---------------------------
def get_all_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data
