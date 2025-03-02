import sqlite3
import pandas as pd
import argparse
import os
from pathlib import Path

def validate_database(db_path, csv_path=None):
    """
    Validate the SQLite database:
    1. Check if tables exist
    2. Check for data integrity
    3. Verify SongFlag is set to TRUE for all records
    4. Verify CSV data if provided
    """
    if not Path(db_path).exists():
        print(f"Error: Database file not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== Validating database: {db_path} ===")
        
        # 1. Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        if 'Rankings' not in table_names:
            print(f"Error: Rankings table not found")
            return False
        else:
            print(f"✓ Rankings table exists")
        
        # 2. Check for data integrity
        # Check if Rankings table has data
        cursor.execute("SELECT COUNT(*) FROM Rankings")
        rankings_count = cursor.fetchone()[0]
        print(f"✓ Rankings table contains {rankings_count} records")
        
        if rankings_count == 0:
            print("Warning: Rankings table is empty")
            return False
        
        # 3. Verify column structure
        cursor.execute("PRAGMA table_info(Rankings)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_columns = [
            'SubjectID', 'Rank', 
            'Driver1', 'Driver2', 'Driver3', 'Driver4', 'Driver5', 
            'Driver6', 'Driver7', 'Driver8', 'Driver9', 'Driver10', 
            'Driver11', 'Driver12', 'Driver13', 'Driver14', 'Driver15', 
            'Driver16', 'Driver17', 'UpdateDT', 'SongFlag'
        ]
        
        missing_columns = [col for col in required_columns if col not in column_names]
        if missing_columns:
            print(f"Error: Missing required columns in Rankings table: {', '.join(missing_columns)}")
            return False
        else:
            print(f"✓ Rankings table has all required columns")
        
        # 4. Verify SongFlag is TRUE for all records
        cursor.execute("SELECT COUNT(*) FROM Rankings WHERE SongFlag = 1")
        true_flag_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Rankings WHERE SongFlag IS NULL OR SongFlag = 0")
        false_flag_count = cursor.fetchone()[0]
        
        if true_flag_count == rankings_count:
            print(f"✓ All {true_flag_count} records have SongFlag set to TRUE")
        else:
            print(f"Error: Expected all records to have SongFlag=TRUE, but found {false_flag_count} records with FALSE or NULL")
        
        # 5. Verify against CSV if provided
        if csv_path and os.path.exists(csv_path):
            try:
                csv_data = pd.read_csv(csv_path)
                print(f"✓ Successfully read CSV file with {len(csv_data)} records")
                
                # Check if all SubjectIDs from CSV exist in database
                subject_ids = set(csv_data['SubjectID'])
                missing_subjects = []
                for subject_id in subject_ids:
                    cursor.execute("SELECT COUNT(*) FROM Rankings WHERE SubjectID = ?", (subject_id,))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        missing_subjects.append(subject_id)
                
                if missing_subjects:
                    print(f"Warning: {len(missing_subjects)} SubjectIDs from CSV not found in database")
                else:
                    print("✓ All SubjectIDs from CSV found in database")
                
                # Check record count
                if len(csv_data) == rankings_count:
                    print(f"✓ Record count matches: {rankings_count} records in database, {len(csv_data)} in CSV")
                else:
                    print(f"Warning: Record count mismatch: {rankings_count} records in database, {len(csv_data)} in CSV")
                
            except Exception as e:
                print(f"Error validating against CSV: {str(e)}")
        
        # Show sample data
        cursor.execute("SELECT SubjectID, Rank, UpdateDT, SongFlag FROM Rankings LIMIT 5")
        sample = cursor.fetchall()
        
        if sample:
            print("\nSample data from database:")
            for row in sample:
                print(f"  SubjectID: {row[0]}, Rank: {row[1]}, UpdateDT: {row[2]}, SongFlag: {'TRUE' if row[3] else 'FALSE'}")
        
        print("\n✓ Database validation completed successfully")
        return True
        
    except sqlite3.Error as e:
        print(f"SQLite error: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate SQLite database with ranking data')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
    parser.add_argumen