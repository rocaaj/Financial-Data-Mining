import sqlite3
import pandas as pd

# Path to the SQLite database
DB_PATH = "datawarehouse.db"

def connect_db(db_path):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print("✅ Connected to database.")
        return conn
    except sqlite3.Error as e:
        print("❌ Connection failed:", e)
        return None

def list_tables(conn):
    """List all tables in the database."""
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    cursor = conn.cursor()
    cursor.execute(query)
    tables = [row[0] for row in cursor.fetchall()]
    print(f"📋 Tables found: {tables}")
    return tables

def preview_table(conn, table_name, limit=5):
    """Preview the first few rows of a table."""
    query = f"SELECT * FROM {table_name} LIMIT {limit};"
    df = pd.read_sql_query(query, conn)
    print(f"\n🔍 Preview of '{table_name}':")
    print(df)
    return df

def main():
    conn = connect_db(DB_PATH)
    if not conn:
        return

    tables = list_tables(conn)
    
    for table in tables:
        preview_table(conn, table)

    conn.close()
    print("🔒 Connection closed.")

if __name__ == "__main__":
    main()
