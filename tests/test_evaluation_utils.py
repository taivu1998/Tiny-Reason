import os
import tempfile
import unittest

import torch

from src.evaluation_utils import (
    decode_generated_text,
    extract_final_answer,
    normalize_answer,
    parse_ground_truth_answer,
    save_evaluation_report,
)


class FakeTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=False):
        del skip_special_tokens
        decoded = []
        for sequence in sequences.tolist():
            decoded.append("".join(chr(token) for token in sequence))
        return decoded


class EvaluationUtilsTests(unittest.TestCase):
    def test_normalize_answer_handles_units_and_number_words(self):
        self.assertEqual(normalize_answer("$1,200 dollars"), "1200")
        self.assertEqual(normalize_answer("Ten"), "10")
        self.assertEqual(normalize_answer("42.0"), "42")

    def test_extract_final_answer_returns_none_when_marker_missing(self):
        self.assertIsNone(extract_final_answer("No marker here"))
        self.assertEqual(
            extract_final_answer("**Final Answer:** 72<|im_end|>"),
            "72",
        )
        self.assertEqual(extract_final_answer("72"), "72")
        self.assertEqual(extract_final_answer("The answer is 72"), "72")
        self.assertEqual(
            extract_final_answer("<think>\n1+1=2\n</think>\nThe answer is 2."),
            "2",
        )

    def test_parse_ground_truth_answer_splits_once(self):
        self.assertEqual(parse_ground_truth_answer("reasoning #### 9"), "9")
        self.assertEqual(parse_ground_truth_answer("a #### b #### c"), "b #### c")
        self.assertIsNone(parse_ground_truth_answer("missing delimiter"))

    def test_decode_generated_text_ignores_prompt_tokens(self):
        tokenizer = FakeTokenizer()
        outputs = torch.tensor([[112, 114, 111, 109, 112, 116, 79, 75]])

        decoded = decode_generated_text(tokenizer, outputs, prompt_length=6)

        self.assertEqual(decoded, "OK")

    def test_save_evaluation_report_writes_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "report.json")
            saved_path = save_evaluation_report({"accuracy": 50.0}, report_path)

            self.assertEqual(saved_path, report_path)
            self.assertTrue(os.path.exists(report_path))


if __name__ == "__main__":
    unittest.main()
