import unittest
import sqlite3
import pandas as pd
from io import StringIO
from datetime import datetime
from data.seed_db import create_table, format_update_dt, import_csv_to_sqlite, load_sqlite_to_dataframe, insert_into_rankings

class TestSQLiteImportModule(unittest.TestCase):
    
    def setUp(self):
        """Set up an in-memory SQLite database for testing."""
        self.conn = sqlite3.connect(':memory:')
        create_table(self.conn)
    
    def tearDown(self):
        """Close the database connection after each test."""
        self.conn.close()
    
    def test_create_table(self):
        """Test that the Rankings table is created successfully."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Rankings'")
        self.assertIsNotNone(cursor.fetchone())
    
    def test_format_update_dt(self):
        """Test date formatting function."""
        input_date = "3/8/2025 14:30"
        expected_output = "2025-03-08 14:30:00"
        self.assertEqual(format_update_dt(input_date), expected_output)
    
    def test_insert_into_rankings(self):
        """Test inserting a single record into the Rankings table."""
        insert_into_rankings(self.conn, 1, 10, ["DriverA", "DriverB"], "2025-03-08 14:30:00")
        df = pd.read_sql("SELECT * FROM Rankings", self.conn)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['SubjectID'], 1)
        self.assertEqual(df.iloc[0]['Rank'], 10)
        self.assertEqual(df.iloc[0]['Driver1'], "DriverA")
        self.assertEqual(df.iloc[0]['Driver2'], "DriverB")
        self.assertEqual(df.iloc[0]['UpdateDT'], "2025-03-08 14:30:00")
    
    def test_import_csv_to_sqlite(self):
        """Test importing CSV data into SQLite."""
        csv_data = """SubjectID,Rank,Driver1,UpdateDT\n1,5,DriverX,3/8/2025 14:30"""
        csv_file = StringIO(csv_data)
        df = pd.read_csv(csv_file, dtype=str)
        df['UpdateDT'] = df['UpdateDT'].apply(format_update_dt)
        df.to_sql('Rankings', self.conn, if_exists='append', index=False)
        df_sql = pd.read_sql("SELECT * FROM Rankings", self.conn)
        self.assertEqual(len(df_sql), 1)
        self.assertEqual(df_sql.iloc[0]['Driver1'], "DriverX")
    
    def test_load_sqlite_to_dataframe(self):
        """Test loading SQLite data into a Pandas DataFrame."""
        insert_into_rankings(self.conn, 2, 3, ["DriverY"], "2025-03-08 14:40:00")
        df = load_sqlite_to_dataframe(':memory:')
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['Driver1'], "DriverY")

if __name__ == '__main__':
    unittest.main()
