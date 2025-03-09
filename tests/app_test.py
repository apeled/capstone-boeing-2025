import unittest
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the functions to test - assuming they're in a module called 'prediction_utils'
# If your functions are in a different module, update this import
import jake_app_copy  as pu

class TestGetSubjectHistory(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.reference_df = pd.DataFrame({
            'SubjectID': [1, 1, 2],
            'Rank': [10, 15, 20],
            'UpdateDT': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-01-10'])
        })
        
    def test_get_subject_history_found(self):
        # Test when subject is found
        history = pu.get_subject_history(1, self.reference_df)
        self.assertIsNotNone(history)
        self.assertEqual(history['previous_rank'], 15)  # Should get most recent (Jan 15)
        self.assertEqual(history['num_records'], 2)
        self.assertEqual(history['avg_rank'], 12.5)
        self.assertEqual(history['min_rank'], 10)
        self.assertEqual(history['max_rank'], 15)
        
    def test_get_subject_history_not_found(self):
        # Test when subject is not found
        history = pu.get_subject_history(999, self.reference_df)
        self.assertIsNone(history)
        
    def test_get_subject_history_empty_dataframe(self):
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        history = pu.get_subject_history(1, empty_df)
        self.assertIsNone(history)
        
    def test_get_subject_history_missing_columns(self):
        # Test with dataframe missing required columns
        df_no_cols = pd.DataFrame({'OtherCol': [1, 2, 3]})
        history = pu.get_subject_history(1, df_no_cols)
        self.assertIsNone(history)

class TestEngineerFeatures(unittest.TestCase):
    def setUp(self):
        # Mock globals that would be defined elsewhere
        self.original_driver_columns = getattr(pu, 'driver_columns', None)
        self.original_scaler = getattr(pu, 'scaler', None)
        self.original_model_path = getattr(pu, 'MODEL_PATH', None)
        
        # Set up test data
        pu.driver_columns = ['Driver1', 'Driver2', 'Driver3']
        pu.MODEL_PATH = '/tmp/mock_model_path'
        pu.scaler = MagicMock()
        pu.scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Create test dataframes
        self.input_df = pd.DataFrame({
            'SubjectID': ['test1'],
            'Driver1': ['factor1'],
            'Driver2': ['factor2'],
            'Driver3': [None],
            'PrevRank': [25]
        })
        
        self.reference_df = pd.DataFrame({
            'SubjectID': ['ref1', 'ref2'],
            'Rank': [10, 20],
            'UpdateDT': pd.to_datetime(['2023-01-01', '2023-01-15']),
            'Driver1': ['factor1', 'factor3'],
            'Driver2': ['factor2', 'factor4'],
            'Driver3': ['factor5', 'factor6']
        })
        
    def tearDown(self):
        # Restore original globals
        if self.original_driver_columns is not None:
            pu.driver_columns = self.original_driver_columns
        else:
            delattr(pu, 'driver_columns')
            
        if self.original_scaler is not None:
            pu.scaler = self.original_scaler
        else:
            delattr(pu, 'scaler')
            
        if self.original_model_path is not None:
            pu.MODEL_PATH = self.original_model_path
        else:
            delattr(pu, 'MODEL_PATH')
    
    def test_engineer_features_basic(self):
        # Test basic functionality
        processed_df, processed_df_onehot = pu.engineer_features(self.input_df, self.reference_df)
        
        # Check that positional dataframe has required columns
        self.assertIn('factor1_Position', processed_df.columns)
        self.assertIn('factor2_Position', processed_df.columns)
        self.assertEqual(processed_df['factor1_Position'].iloc[0], 1)  # Driver1 position
        self.assertEqual(processed_df['factor2_Position'].iloc[0], 2)  # Driver2 position
        
        # Check one-hot encoding
        self.assertIn('Driver1_factor1', processed_df_onehot.columns)
        self.assertIn('Driver2_factor2', processed_df_onehot.columns)
        self.assertEqual(processed_df_onehot['Driver1_factor1'].iloc[0], 1)
        self.assertEqual(processed_df_onehot['Driver2_factor2'].iloc[0], 1)
        
        # Check time features
        self.assertIn('Year', processed_df.columns)
        self.assertIn('Month', processed_df.columns)
        self.assertIn('DayOfMonth', processed_df.columns)
        self.assertIn('DayOfWeek', processed_df.columns)
        
    @patch('os.path.exists')
    @patch('pickle.load')
    def test_engineer_features_with_feature_columns(self, mock_pickle_load, mock_path_exists):
        # Test loading feature columns
        mock_path_exists.return_value = True
        mock_pickle_load.return_value = ['Driver1_factor1', 'Driver1_factor3', 'Driver2_factor2', 'Custom_Feature']
        
        processed_df, processed_df_onehot = pu.engineer_features(self.input_df, self.reference_df)
        
        # Check that missing features were added
        self.assertIn('Custom_Feature', processed_df_onehot.columns)
        self.assertEqual(processed_df_onehot['Custom_Feature'].iloc[0], 0)
        
    def test_engineer_features_missing_subjectid(self):
        # Test with missing SubjectID
        input_df_no_id = self.input_df.copy()
        input_df_no_id['SubjectID'] = ""
        
        processed_df, processed_df_onehot = pu.engineer_features(input_df_no_id, self.reference_df)
        
        # Should get a default ID
        self.assertEqual(processed_df['SubjectID'].iloc[0], 'new_input')

class TestGetModelFeatures(unittest.TestCase):
    def test_get_model_features_onehot(self):
        self.assertEqual(pu.get_model_features('Random Forest One-Hot'), 'onehot')
        self.assertEqual(pu.get_model_features('XGBoost One-Hot Encoded'), 'onehot')
        
    def test_get_model_features_positional(self):
        self.assertEqual(pu.get_model_features('Linear Regression Positional'), 'positional')
        self.assertEqual(pu.get_model_features('LGBM Positional Features'), 'positional')
        
    def test_get_model_features_pca(self):
        self.assertEqual(pu.get_model_features('SVM with PCA'), 'onehot')
        
    def test_get_model_features_default(self):
        self.assertEqual(pu.get_model_features('Unknown Model Type'), 'onehot')

class TestGetFeatureImportances(unittest.TestCase):
    def setUp(self):
        self.feature_names = ['feature1', 'feature2', 'feature3']
        
    def test_feature_importances_attribute(self):
        # Test with model having feature_importances_ attribute
        model = MagicMock()
        model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        result = pu.get_feature_importances(model, self.feature_names)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['Feature'].tolist(), ['feature1', 'feature2', 'feature3'])
        self.assertEqual(result['Importance'].tolist(), [0.5, 0.3, 0.2])
        
    def test_coef_attribute(self):
        # Test with model having coef_ attribute
        model = MagicMock()
        model.feature_importances_ = None
        model.coef_ = np.array([0.5, -0.3, 0.2])
        
        result = pu.get_feature_importances(model, self.feature_names)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['Feature'].tolist(), ['feature1', 'feature3', 'feature2'])
        self.assertEqual(result['Importance'].tolist(), [0.5, 0.2, 0.3])
        
    def test_dictionary_model(self):
        # Test with dictionary model
        model = {
            'feature_importances_': np.array([0.5, 0.3, 0.2])
        }
        
        result = pu.get_feature_importances(model, self.feature_names)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['Feature'].tolist(), ['feature1', 'feature2', 'feature3'])
        
    def test_nested_model(self):
        # Test with nested model in dictionary
        nested_model = MagicMock()
        nested_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        model = {
            'estimator': nested_model
        }
        
        result = pu.get_feature_importances(model, self.feature_names)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['Feature'].tolist(), ['feature1', 'feature2', 'feature3'])
        
    def test_length_mismatch(self):
        # Test with length mismatch between importances and feature names
        model = MagicMock()
        model.feature_importances_ = np.array([0.5, 0.3])  # Only 2 values
        
        result = pu.get_feature_importances(model, self.feature_names)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Should truncate to shorter length
        self.assertEqual(result['Feature'].tolist(), ['feature1', 'feature2'])
        
class TestFindModelFiles(unittest.TestCase):
    @patch('os.listdir')
    def test_find_model_files(self, mock_listdir):
        # Set up test environment
        pu.MODEL_PATH = '/test/path'
        
        # Mock the model summary DataFrame
        global model_summary
        original_model_summary = globals().get('model_summary', None)
        model_summary = pd.DataFrame({
            'model_name': ['Random Forest Model', 'XGBoost Model'],
            'filename': ['random_forest.pkl', 'xgboost.pkl']
        })
        
        # Mock os.listdir to return some test files
        mock_listdir.return_value = ['random_forest.pkl', 'xgboost.pkl', 'pca_model.pkl', 'feature_columns.pkl', 'other_file.txt']
        
        # Call the function
        result = pu.find_model_files()
        
        # Check results
        self.assertEqual(len(result), 2)
        self.assertIn('Random Forest Model', result)
        self.assertIn('XGBoost Model', result)
        self.assertEqual(result['Random Forest Model'], '/test/path/random_forest.pkl')
        
        # Restore globals
        if original_model_summary is not None:
            model_summary = original_model_summary
        else:
            del globals()['model_summary']
            
    @patch('os.listdir')
    def test_find_model_files_no_summary(self, mock_listdir):
        # Test when model_summary is not available
        pu.MODEL_PATH = '/test/path'
        
        # Remove model_summary if it exists
        global model_summary
        original_model_summary = globals().get('model_summary', None)
        if 'model_summary' in globals():
            del globals()['model_summary']
        
        # Mock os.listdir to return some test files
        mock_listdir.return_value = ['random_forest.pkl', 'xgboost.pkl']
        
        # Call the function
        result = pu.find_model_files()
        
        # Check results - should use filenames as model names
        self.assertEqual(len(result), 2)
        self.assertIn('Random Forest', result)
        self.assertIn('Xgboost', result)
        
        # Restore globals
        if original_model_summary is not None:
            model_summary = original_model_summary
            
    @patch('os.listdir')
    def test_find_model_files_file_not_found(self, mock_listdir):
        # Test handling of FileNotFoundError
        pu.MODEL_PATH = '/test/path'
        mock_listdir.side_effect = FileNotFoundError
        
        result = pu.find_model_files()
        
        self.assertEqual(len(result), 0)

class TestLoadModel(unittest.TestCase):
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_direct(self, mock_open, mock_pickle_load):
        # Test loading a model directly
        mock_model = MagicMock()
        mock_pickle_load.return_value = mock_model
        
        result = pu.load_model('/path/to/model.pkl')
        
        self.assertEqual(result, mock_model)
        
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_from_dict_with_model_key(self, mock_open, mock_pickle_load):
        # Test loading a model from a dictionary with 'model' key
        mock_model = MagicMock()
        mock_pickle_load.return_value = {'model': mock_model}
        
        result = pu.load_model('/path/to/model.pkl')
        
        self.assertEqual(result, mock_model)
        
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_from_dict_with_estimator_key(self, mock_open, mock_pickle_load):
        # Test loading a model from a dictionary with 'estimator' key
        mock_model = MagicMock()
        mock_pickle_load.return_value = {'estimator': mock_model}
        
        result = pu.load_model('/path/to/model.pkl')
        
        self.assertEqual(result, mock_model)
        
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_from_dict_with_model_like_object(self, mock_open, mock_pickle_load):
        # Test loading a model from a dictionary with a model-like object
        mock_model = MagicMock()
        mock_model.predict = MagicMock()  # Add predict method
        mock_pickle_load.return_value = {'custom_key': mock_model}
        
        result = pu.load_model('/path/to/model.pkl')
        
        self.assertEqual(result, mock_model)
        
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_file_not_found(self, mock_open, mock_pickle_load):
        # Test handling of FileNotFoundError
        mock_open.side_effect = FileNotFoundError
        
        result = pu.load_model('/path/to/model.pkl')
        
        self.assertIsNone(result)
        
    def test_load_model_empty_path(self):
        # Test with empty model path
        result = pu.load_model('')
        
        self.assertIsNone(result)

class TestMakePredictionWithModel(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2, 3]])
        
    def test_make_prediction_with_model_none(self):
        # Test with None model
        with self.assertRaises(ValueError):
            pu.make_prediction_with_model(None, self.X)
            
    def test_make_prediction_with_normal_model(self):
        # Test with normal model having predict method
        model = MagicMock()
        model.predict.return_value = np.array([42])
        
        result = pu.make_prediction_with_model(model, self.X)
        
        self.assertEqual(result.tolist(), [42])
        model.predict.assert_called_once_with(self.X)
        
    def test_make_prediction_with_dict_model(self):
        # Test with dictionary model containing 'model' key
        inner_model = MagicMock()
        inner_model.predict.return_value = np.array([42])
        model = {'model': inner_model}
        
        result = pu.make_prediction_with_model(model, self.X)
        
        self.assertEqual(result.tolist(), [42])
        inner_model.predict.assert_called_once_with(self.X)
        
    def test_make_prediction_with_dict_model_direct_predict(self):
        # Test with dictionary model having predict method directly
        model = MagicMock()
        model.predict.return_value = np.array([42])
        model_dict = {'some_key': 'some_value'}
        # Add the predict method to the dictionary
        model_dict.predict = model.predict
        
        result = pu.make_prediction_with_model(model_dict, self.X)
        
        self.assertEqual(result.tolist(), [42])
        model.predict.assert_called_once_with(self.X)
        
    def test_make_prediction_with_dict_predictions(self):
        # Test with dictionary containing precomputed predictions
        model = {'predictions': np.array([42])}
        
        result = pu.make_prediction_with_model(model, self.X)
        
        self.assertEqual(result.tolist(), [42])
        
    def test_make_prediction_with_pipeline(self):
        # Test with sklearn Pipeline
        final_estimator = MagicMock()
        final_estimator.predict.return_value = np.array([42])
        
        model = MagicMock()
        model.steps = [('preprocessing', MagicMock()), ('estimator', final_estimator)]
        
        result = pu.make_prediction_with_model(model, self.X)
        
        self.assertEqual(result.tolist(), [42])
        final_estimator.predict.assert_called_once_with(self.X)
        
    def test_make_prediction_with_unsupported_model(self):
        # Test with a model that doesn't support prediction
        model = "Not a model"
        
        with self.assertRaises(ValueError):
            pu.make_prediction_with_model(model, self.X)

if __name__ == '__main__':
    unittest.main()