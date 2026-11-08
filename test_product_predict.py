import unittest
from unittest.mock import MagicMock
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from product_predict import validate_inputs, predict_one


class TestProductPredict(unittest.TestCase):

    def setUp(self):
        self.valid_input = {
            "age": 35,
            "job": "admin",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 1500,
            "housing_loan": "yes",
            "personal_loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "duration": 300,
            "campaign": 2,
            "pdays": 100,
            "previous": 1,
            "poutcome": "success"
        }

    def test_valid_input(self):
        result = validate_inputs(self.valid_input)
        self.assertEqual(result["age"], 35.0)
        self.assertEqual(result["balance"], 1500.0)
        self.assertEqual(result["job"], "admin")

    def test_missing_numeric_feature(self):
        data = self.valid_input.copy()
        del data["age"]

        with self.assertRaises(ValueError):
            validate_inputs(data)

    def test_invalid_age(self):
        data = self.valid_input.copy()
        data["age"] = 15

        with self.assertRaises(ValueError):
            validate_inputs(data)

    def test_negative_duration(self):
        data = self.valid_input.copy()
        data["duration"] = -10

        with self.assertRaises(ValueError):
            validate_inputs(data)

    def test_prediction_output(self):
        model = MagicMock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.10, 0.90]])

        result = predict_one(model, self.valid_input)

        self.assertEqual(result["status"], "APPROVED")
        self.assertEqual(result["predicted_class"], 1)
        self.assertEqual(result["label"], "Loan Approved")
        self.assertEqual(result["approval_probability"], 90.0)
        self.assertEqual(result["default_risk"], 10.0)


if __name__ == "__main__":
    unittest.main()