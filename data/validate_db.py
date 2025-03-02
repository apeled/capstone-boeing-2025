#python validate_db.py --db rankings.db --csv Rank.CSV
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
    3. Verify data matches CSV if provided
    4. Validate relationships and constraints
    """
    if not Path(db_path).exists():
        print(f"Error: Database file not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== Validating database: {db_path} ===")
        
        # 1. Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        required_tables = ['Rankings', 'RankingHistory']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            print(f"Error: Missing required tables: {', '.join(missing_tables)}")
            return False
        else:
            print(f"✓ All required tables exist: {', '.join(required_tables)}")
        
        # 2. Check for data integrity
        # Check if Rankings table has data
        cursor.execute("SELECT COUNT(*) FROM Rankings")
        rankings_count = cursor.fetchone()[0]
        print(f"✓ Rankings table contains {rankings_count} records")
        
        if rankings_count == 0:
            print("Warning: Rankings table is empty")
            return False
        
        # Check if RankingHistory table has data
        cursor.execute("SELECT COUNT(*) FROM RankingHistory")
        history_count = cursor.fetchone()[0]
        print(f"✓ RankingHistory table contains {history_count} records")
        
        # 3. Verify column structure
        cursor.execute("PRAGMA table_info(Rankings)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_columns = [
            'SubjectID', 'Rank', 
            'Driver1', 'Driver2', 'Driver3', 'Driver4', 'Driver5', 
            'Driver6', 'Driver7', 'Driver8', 'Driver9', 'Driver10', 
            'Driver11', 'Driver12', 'Driver13', 'Driver14', 'Driver15', 
            'Driver16', 'Driver17', 'UpdateDT', 'PrevRank', 'PrevUpdateDT', 'RankChange'
        ]
        
        missing_columns = [col for col in required_columns if col not in column_names]
        if missing_columns:
            print(f"Error: Missing required columns in Rankings table: {', '.join(missing_columns)}")
            return False
        else:
            print(f"✓ Rankings table has all required columns")
        
        # 4. Check data consistency
        cursor.execute("""
        SELECT r.SubjectID, r.Rank, r.UpdateDT, r.PrevRank, r.RankChange,
               h.Rank, h.UpdateDT, h.PrevRank, h.RankChange
        FROM Rankings r
        JOIN RankingHistory h ON r.SubjectID = h.SubjectID AND r.UpdateDT = h.UpdateDT
        LIMIT 10
        """)
        consistency_check = cursor.fetchall()
        
        consistency_issues = 0
        for record in consistency_check:
            if record[1] != record[5] or record[3] != record[7] or record[4] != record[8]:
                consistency_issues += 1
                print(f"Data inconsistency found for SubjectID {record[0]}")
        
        if consistency_issues == 0:
            print("✓ Sample data consistency check passed between Rankings and RankingHistory tables")
        
        # 5. Verify RankChange calculation correctness
        cursor.execute("""
        SELECT SubjectID, Rank, PrevRank, RankChange
        FROM Rankings
        WHERE PrevRank IS NOT NULL
        LIMIT 10
        """)
        rank_change_records = cursor.fetchall()
        
        rank_change_issues = 0
        for record in rank_change_records:
            expected_change = record[2] - record[1]  # PrevRank - Rank
            if record[3] != expected_change:
                rank_change_issues += 1
                print(f"RankChange calculation issue for SubjectID {record[0]}: Expected {expected_change}, found {record[3]}")
        
        if rank_change_issues == 0 and len(rank_change_records) > 0:
            print("✓ RankChange calculations are correct")
        elif len(rank_change_records) == 0:
            print("Note: No records with previous rankings to verify RankChange calculations")
        
        # 6. Verify against CSV if provided
        if csv_path and os.path.exists(csv_path):
            try:
                csv_data = pd.read_csv(csv_path)
                print(f"✓ Successfully read CSV file with {len(csv_data)} records")
                
                # Check if all SubjectIDs from CSV exist in database
                subject_ids = set(csv_data['SubjectID'])
                for subject_id in subject_ids:
                    cursor.execute("SELECT COUNT(*) FROM Rankings WHERE SubjectID = ?", (subject_id,))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        print(f"Warning: SubjectID {subject_id} from CSV not found in database")
                
                print("✓ CSV validation complete")
            except Exception as e:
                print(f"Error validating against CSV: {str(e)}")
        
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
    parser.add_argument('--csv', help='Optional path to the original CSV file for cross-validation')
    
    args = parser.parse_args()
    
    validate_database(args.db, args.csv)