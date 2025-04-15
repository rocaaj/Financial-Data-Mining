import sqlite3
import pandas as pd
import os

# Path to the SQLite database
DB_PATH = "datawarehouse.db"
CSV_EXPORT_DIR = "output"

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

def describe_table_schema(conn, table_name):
    """Print column data types and basic stats for a table."""
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)

    print(f"\n🧬 Schema + Info for '{table_name}':")
    print(df.dtypes)
    print("\n📊 Summary Statistics:")
    print(df.describe(include='all'))
    
    print("\n❓ Null Counts:")
    print(df.isnull().sum())

def export_tables_to_csv(conn, tables, export_dir=CSV_EXPORT_DIR):
    """Export all tables to individual CSV files."""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        csv_path = os.path.join(export_dir, f"{table}.csv")
        df.to_csv(csv_path, index=False)
        print(f"📁 Exported '{table}' to {csv_path}")

def main():
    conn = connect_db(DB_PATH)
    if not conn:
        return

    tables = list_tables(conn)

    for table in tables:
        preview_table(conn, table)
        describe_table_schema(conn, table)

    export_tables_to_csv(conn, tables)

    conn.close()
    print("🔒 Connection closed.")

if __name__ == "__main__":
    main()
