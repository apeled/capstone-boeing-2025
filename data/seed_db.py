import csv
import sqlite3
import datetime
import os
from pathlib import Path

def create_database(db_path):
    """Create SQLite database and tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create main rankings table with only the specified columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Rankings (
        SubjectID TEXT,
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
        SongFlag BOOLEAN,
        PRIMARY KEY (SubjectID, UpdateDT)
    )
    ''')
    
    conn.commit()
    return conn

def seed_database(csv_path, db_path):
    """Seed the database with data from CSV file"""
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    conn = create_database(db_path)
    cursor = conn.cursor()
    
    # Read CSV file
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Process each row
        for row in csv_reader:
            subject_id = row['SubjectID']
            current_rank = int(row['Rank'])
            update_dt = row['UpdateDT']
            
            # Format date properly if it's in ####### format
            if update_dt.startswith('#') and update_dt.endswith('#'):
                # Assuming this is a placeholder and we should use current datetime
                update_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Insert into Rankings table with SongFlag set to true (1 in SQLite)
            cursor.execute('''
            INSERT INTO Rankings (
                SubjectID, Rank, 
                Driver1, Driver2, Driver3, Driver4, Driver5, 
                Driver6, Driver7, Driver8, Driver9, Driver10, 
                Driver11, Driver12, Driver13, Driver14, Driver15, 
                Driver16, Driver17, UpdateDT, SongFlag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subject_id, current_rank,
                row['Driver1'], row['Driver2'], row['Driver3'], row['Driver4'], row['Driver5'],
                row['Driver6'], row['Driver7'], row['Driver8'], row['Driver9'], row['Driver10'],
                row['Driver11'], row['Driver12'], row['Driver13'], row['Driver14'], row['Driver15'],
                row['Driver16'], row['Driver17'], update_dt, 1  # 1 means TRUE for SongFlag
            ))
    
    conn.commit()
    conn.close()
    print(f"Database successfully seeded from {csv_path} to {db_path} with SongFlag set to TRUE")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed SQLite database with CSV ranking data')
    parser.add_argument('--csv', required=True, help='Path to the CSV file')
    parser.add_argument('--db', default='rankings.db', help='Path to SQLite database (default: rankings.db)')
    
    args = parser.parse_args()
    
    seed_database(args.csv, args.db)