import unittest
from src.product_predict import validate_inputs

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid_applicant = {
            "age": "35", "job": "management", "marital": "married",
            "education": "tertiary", "default": "no", "balance": "4500.0",
            "housing_loan": "no", "personal_loan": "no", "contact": "cellular",
            "day": "15", "month": "may", "duration": "250.0",
            "campaign": "1", "pdays": "-1", "previous": "0", "poutcome": "unknown",
        }

    def test_normal(self):
        clean = validate_inputs(self.valid_applicant)
        self.assertEqual(clean["age"], 35.0)

    def test_age_boundaries(self):
        self.valid_applicant["age"] = "18"
        self.assertEqual(validate_inputs(self.valid_applicant)["age"], 18.0)
        self.valid_applicant["age"] = "120"
        self.assertEqual(validate_inputs(self.valid_applicant)["age"], 120.0)

    def test_out_of_range_age(self):
        self.valid_applicant["age"] = "15"
        with self.assertRaises(ValueError):
            validate_inputs(self.valid_applicant)

    def test_invalid_numeric_text(self):
        self.valid_applicant["balance"] = "abc"
        with self.assertRaises(ValueError):
            validate_inputs(self.valid_applicant)

    def test_negative_duration(self):
        self.valid_applicant["duration"] = "-50"
        with self.assertRaises(ValueError):
            validate_inputs(self.valid_applicant)

    def test_missing_feature(self):
        self.valid_applicant["job"] = ""
        with self.assertRaises(ValueError):
            validate_inputs(self.valid_applicant)

if __name__ == "__main__":
    unittest.main()