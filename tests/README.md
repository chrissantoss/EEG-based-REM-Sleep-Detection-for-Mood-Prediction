# Tests for EEG-based REM Sleep Detection

This directory contains unit tests for the EEG-based REM sleep detection project. The tests verify the functionality of various components of the pipeline, including data processing, feature extraction, and model training.

## Running the Tests

To run all tests, use the following command from the project root directory:

```bash
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/test_pipeline.py
```

To run a specific test class:

```bash
python -m pytest tests/test_pipeline.py::TestDataProcessing
```

To run a specific test method:

```bash
python -m pytest tests/test_pipeline.py::TestDataProcessing::test_segment_eeg_data
```

## Test Coverage

The tests cover the following components:

1. **Data Download**: Tests the functionality of downloading and generating datasets.
2. **Data Processing**: Tests the EEG data processing pipeline, including filtering, segmentation, and feature extraction.
3. **Feature Extraction**: Tests the feature selection and preparation for model training.
4. **Model Training**: Tests the model training and evaluation functionality.

## Adding New Tests

When adding new functionality to the project, please also add corresponding tests. Follow these guidelines:

1. Create test classes that inherit from `unittest.TestCase`.
2. Use descriptive method names that start with `test_`.
3. Include assertions to verify the expected behavior.
4. Use setUp and tearDown methods for common setup and cleanup tasks.

Example:

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup code
        pass
        
    def test_new_feature(self):
        # Test code
        result = new_feature_function()
        self.assertEqual(result, expected_result)
        
    def tearDown(self):
        # Cleanup code
        pass 