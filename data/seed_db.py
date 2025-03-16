"""
SQLite Data Import Module

This module creates an SQLite database table `Rankings` and imports data from a CSV file.

Features:
- Creates the `Rankings` table if it does not exist.
- Reads `Rank.csv`, formats the `UpdateDT` column, and inserts data.
- Drops duplicate rows before importing.
- Sets `OriginalDataFlag` to `True` and `Model` to `NULL`.
- Provides a function to load the database into a Pandas DataFrame.
"""

import sqlite3
import pandas as pd
from datetime import datetime

def create_table(conn):
    """Creates the Rankings table in the SQLite database if it does not exist."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Rankings (
            SubjectID INTEGER,
            Rank INTEGER,
            Driver1 TEXT,
            Driver2 TEXT,
            Driver3 TEXT,
            Driver4 TEXT,
            Driver5 TEXT,
            Driver6 TEXT,
            Driver7 TEXT,
            Driver8 TEXT,
            Driver9 TEXT,
            Driver10 TEXT,
            Driver11 TEXT,
            Driver12 TEXT,
            Driver13 TEXT,
            Driver14 TEXT,
            Driver15 TEXT,
            Driver16 TEXT,
            Driver17 TEXT,
            UpdateDT TEXT,
            OriginalDataFlag BOOLEAN,
            Model TEXT,
            PRIMARY KEY (SubjectID, UpdateDT, Model)
        )
    ''')
    conn.commit()

def format_update_dt(date_str):
    """Converts date from 'M/D/YYYY HH:MM' format to 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.strptime(date_str, "%m/%d/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")

def import_csv_to_sqlite(db_file, csv_file):
    """Imports data from a CSV file into the Rankings table in the SQLite database."""
    conn = sqlite3.connect(db_file)
    create_table(conn)

    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, dtype=str)

    # Convert UpdateDT to standard format
    df['UpdateDT'] = df['UpdateDT'].apply(format_update_dt)

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Add missing columns
    df['OriginalDataFlag'] = True  # Set to True
    df['Model'] = None  # Leave blank

    # Insert data into SQLite
    df.to_sql('Rankings', conn, if_exists='append', index=False)

    # Close connection
    conn.close()
    print("Data import completed successfully!")

def load_sqlite_to_dataframe(db_file):
    """Loads the Rankings table from the SQLite database into a Pandas DataFrame."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql("SELECT * FROM Rankings", conn)
    conn.close()
    return df

def insert_into_rankings(conn, subject_id, rank, drivers, update_dt, original_data_flag=True, model=None):
    """
    Inserts a single record into the Rankings table, handling potential uniqueness constraint violations.

    Parameters:
        conn (sqlite3.Connection): SQLite database connection.
        subject_id (int): Subject ID.
        rank (int): Rank value.
        drivers (list of str): List of up to 17 driver names.
        update_dt (str): Date and time in 'YYYY-MM-DD HH:MM:SS' format.
        original_data_flag (bool, optional): Flag indicating original data. Defaults to True.
        model (str, optional): Model name. Defaults to None.
    """
    cursor = conn.cursor()
    
    # Ensure drivers list has exactly 17 elements (fill missing with None)
    drivers = (drivers + [None] * 17)[:17]

    try:
        cursor.execute('''
            INSERT INTO Rankings (
                SubjectID, Rank, Driver1, Driver2, Driver3, Driver4, Driver5, Driver6, Driver7, 
                Driver8, Driver9, Driver10, Driver11, Driver12, Driver13, Driver14, Driver15, Driver16, 
                Driver17, UpdateDT, OriginalDataFlag, Model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (subject_id, rank, *drivers, update_dt, original_data_flag, model))
        
        conn.commit()
        print(f"Inserted record: SubjectID={subject_id}, Rank={rank}, UpdateDT={update_dt}")

    except sqlite3.IntegrityError:
        print(f"Skipping duplicate entry: SubjectID={subject_id}, UpdateDT={update_dt}")

if __name__ == "__main__":
    DB_FILE = "rankings.db"
    CSV_FILE = "Rank.csv"
    import_csv_to_sqlite(DB_FILE, CSV_FILE)
