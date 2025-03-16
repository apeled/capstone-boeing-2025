"""
SQLite Database Validation Script

This script validates the structure and content of the SQLite database created by the seed_db.py module.
It performs the following checks:
- Verifies the database file exists
- Validates the table structure matches the expected schema
- Checks for data integrity (missing values, duplicates, invalid formats)
- Verifies the OriginalDataFlag and Model settings
- Provides a summary report of the validation results
"""

import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime

def validate_database_existence(db_file):
    """Checks if the database file exists."""
    if not os.path.exists(db_file):
        print(f"ERROR: Database file '{db_file}' does not exist.")
        return False
    
    file_size = os.path.getsize(db_file)
    print(f"Database file exists (Size: {file_size/1024:.2f} KB)")
    return True

def validate_table_structure(conn):
    """Validates the structure of the Rankings table."""
    cursor = conn.cursor()
    
    # Check if the Rankings table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Rankings'")
    if not cursor.fetchone():
        print("ERROR: Rankings table does not exist in the database.")
        return False
    
    # Get table schema
    cursor.execute("PRAGMA table_info(Rankings)")
    columns = cursor.fetchall()
    
    # Expected columns with their types
    expected_columns = {
        "SubjectID": "INTEGER",
        "Rank": "INTEGER",
        "Driver1": "TEXT",
        "Driver2": "TEXT",
        "Driver3": "TEXT",
        "Driver4": "TEXT",
        "Driver5": "TEXT",
        "Driver6": "TEXT",
        "Driver7": "TEXT",
        "Driver8": "TEXT",
        "Driver9": "TEXT",
        "Driver10": "TEXT",
        "Driver11": "TEXT",
        "Driver12": "TEXT",
        "Driver13": "TEXT",
        "Driver14": "TEXT",
        "Driver15": "TEXT",
        "Driver16": "TEXT",
        "Driver17": "TEXT",
        "UpdateDT": "TEXT",
        "OriginalDataFlag": "BOOLEAN",
        "Model": "TEXT"
    }
    
    # Validate columns
    actual_columns = {col[1]: col[2] for col in columns}
    missing_columns = set(expected_columns.keys()) - set(actual_columns.keys())
    extra_columns = set(actual_columns.keys()) - set(expected_columns.keys())
    type_mismatches = []
    
    for col, expected_type in expected_columns.items():
        if col in actual_columns and not actual_columns[col].upper().startswith(expected_type):
            type_mismatches.append(f"{col} (Expected: {expected_type}, Actual: {actual_columns[col]})")
    
    # Check primary key
    cursor.execute("PRAGMA table_info(Rankings)")
    columns = cursor.fetchall()
    primary_key_columns = [col[1] for col in columns if col[5] > 0]  # column 5 is the pk flag
    expected_primary_key = ["SubjectID", "UpdateDT", "Model"]
    primary_key_matches = set(primary_key_columns) == set(expected_primary_key)
    
    if missing_columns or extra_columns or type_mismatches or not primary_key_matches:
        if missing_columns:
            print(f"ERROR: Missing columns: {', '.join(missing_columns)}")
        if extra_columns:
            print(f"WARNING: Extra columns: {', '.join(extra_columns)}")
        if type_mismatches:
            print(f"ERROR: Column type mismatches: {', '.join(type_mismatches)}")
        if not primary_key_matches:
            print(f"ERROR: Primary key mismatch. Expected: {expected_primary_key}, Actual: {primary_key_columns}")
        return False
    
    print("Table structure validation: PASSED")
    return True

def validate_data_integrity(conn):
    """Validates the integrity of the data in the Rankings table."""
    df = pd.read_sql("SELECT * FROM Rankings", conn)
    
    print(f"\nData Summary:")
    print(f"- Total records: {len(df)}")
    
    # Check for missing values in required fields
    required_fields = ["SubjectID", "Rank", "UpdateDT"]
    missing_values = {field: df[field].isnull().sum() for field in required_fields}
    
    if any(missing_values.values()):
        print("ERROR: Missing values in required fields:")
        for field, count in missing_values.items():
            if count > 0:
                print(f"  - {field}: {count} missing values")
        validation_passed = False
    else:
        print("- No missing values in required fields")
        validation_passed = True
    
    # Check for duplicate primary keys
    duplicates = df.duplicated(subset=["SubjectID", "UpdateDT", "Model"]).sum()
    if duplicates > 0:
        print(f"ERROR: Found {duplicates} duplicate primary key combinations")
        validation_passed = False
    else:
        print("- No duplicate primary keys")
    
    # Validate date format in UpdateDT
    try:
        df['UpdateDT_parsed'] = pd.to_datetime(df['UpdateDT'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        print("- UpdateDT format validation: PASSED")
    except ValueError:
        invalid_dates = df[~df['UpdateDT'].str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')]['UpdateDT'].tolist()
        print(f"ERROR: Invalid date format in UpdateDT column. Examples: {invalid_dates[:5]}")
        validation_passed = False
    
    # Check OriginalDataFlag values
    if not all(df['OriginalDataFlag'] == True):
        print("ERROR: Not all OriginalDataFlag values are set to True")
        validation_passed = False
    else:
        print("- All OriginalDataFlag values are correctly set to True")
    
    # Check Model values
    if not all(df['Model'].isnull()):
        print("ERROR: Not all Model values are NULL")
        validation_passed = False
    else:
        print("- All Model values are correctly set to NULL")
    
    return validation_passed

def validate_csv_import(db_file, csv_file):
    """Validates that all CSV records were properly imported into the database."""
    if not os.path.exists(csv_file):
        print(f"WARNING: Cannot validate CSV import. CSV file '{csv_file}' does not exist.")
        return True
    
    # Read CSV data
    csv_df = pd.read_csv(csv_file, dtype=str)
    
    # Connect to database
    conn = sqlite3.connect(db_file)
    db_df = pd.read_sql("SELECT * FROM Rankings", conn)
    
    # Compare record counts (accounting for duplicates in CSV that should be dropped)
    csv_unique = csv_df.drop_duplicates()
    csv_count = len(csv_unique)
    db_count = len(db_df)
    
    if db_count < csv_count:
        print(f"ERROR: Database has fewer records ({db_count}) than unique CSV entries ({csv_count})")
        return False
    elif db_count > csv_count:
        print(f"WARNING: Database has more records ({db_count}) than unique CSV entries ({csv_count})")
    else:
        print(f"- Record count matches: {db_count} records")
    
    # Sample validation of a few key fields
    if not csv_df.empty and not db_df.empty:
        # Convert CSV UpdateDT to match database format for comparison
        from datetime import datetime
        csv_df['UpdateDT_formatted'] = csv_df['UpdateDT'].apply(
            lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S") 
            if isinstance(x, str) else x
        )
        
        # Check if all SubjectIDs from CSV exist in database
        csv_subjects = set(csv_df['SubjectID'].astype(str))
        db_subjects = set(db_df['SubjectID'].astype(str))
        missing_subjects = csv_subjects - db_subjects
        
        if missing_subjects:
            print(f"ERROR: {len(missing_subjects)} SubjectIDs from CSV are missing in database")
            print(f"Examples: {list(missing_subjects)[:5]}")
            return False
        else:
            print("- All SubjectIDs from CSV are present in database")
    
    conn.close()
    return True

def run_validation(db_file, csv_file=None):
    """Runs all validation checks and provides a summary report."""
    print("=" * 60)
    print("DATABASE VALIDATION REPORT")
    print("=" * 60)
    print(f"Database file: {db_file}")
    if csv_file:
        print(f"CSV file: {csv_file}")
    print(f"Validation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Check database existence
    if not validate_database_existence(db_file):
        return False
    
    # Connect to the database
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"ERROR: Failed to connect to database: {e}")
        return False
    
    # Run validation checks
    structure_valid = validate_table_structure(conn)
    print("-" * 60)
    
    data_valid = validate_data_integrity(conn)
    print("-" * 60)
    
    import_valid = True
    if csv_file:
        import_valid = validate_csv_import(db_file, csv_file)
        print("-" * 60)
    
    # Overall validation result
    validation_result = structure_valid and data_valid and import_valid
    
    print("VALIDATION SUMMARY:")
    print(f"- Database exists: {'PASSED' if os.path.exists(db_file) else 'FAILED'}")
    print(f"- Table structure: {'PASSED' if structure_valid else 'FAILED'}")
    print(f"- Data integrity: {'PASSED' if data_valid else 'FAILED'}")
    if csv_file:
        print(f"- CSV import: {'PASSED' if import_valid else 'FAILED'}")
    print("-" * 60)
    print(f"OVERALL VALIDATION: {'PASSED' if validation_result else 'FAILED'}")
    print("=" * 60)
    
    conn.close()
    return validation_result

if __name__ == "__main__":
    DB_FILE = "rankings.db"
    CSV_FILE = "Rank.csv"
    validation_passed = run_validation(DB_FILE, CSV_FILE)
