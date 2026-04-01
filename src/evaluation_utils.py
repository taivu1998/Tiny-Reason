from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


FINAL_ANSWER_PATTERN = re.compile(
    r"\*\*Final Answer:\*\*\s*(.*?)(?:<\|im_end\|>|$)", re.DOTALL
)
PLAIN_ANSWER_PATTERN = re.compile(
    r"(?:final answer|answer)\s*(?::|is)\s*([^\n<]+)", re.IGNORECASE
)
NUMERIC_TAIL_PATTERN = re.compile(r"(-?\$?\d[\d,]*(?:\.\d+)?)")
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_answer(ans: str) -> str:
    """Normalizes answers for more forgiving exact-match comparisons."""
    normalized = ans.strip().lower()
    normalized = normalized.replace(",", "").replace(" ", "")
    normalized = re.sub(r"\.0+$", "", normalized)

    for word, digit in NUMBER_WORDS.items():
        normalized = re.sub(rf"^{word}$", digit, normalized)

    normalized = re.sub(r"^\$", "", normalized)
    normalized = re.sub(r"dollars?$", "", normalized)
    normalized = re.sub(r"meters?$", "", normalized)
    normalized = re.sub(r"feet$", "", normalized)
    return normalized.strip()


def extract_final_answer(response: str) -> Optional[str]:
    """Extracts the final answer from a generated completion."""
    match = FINAL_ANSWER_PATTERN.search(response)
    if match is not None:
        return match.group(1).strip()

    cleaned = response.replace("<|im_end|>", " ").strip()
    if not cleaned:
        return None

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    elif "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[1].strip()

    plain_match = PLAIN_ANSWER_PATTERN.search(cleaned)
    if plain_match is not None:
        return plain_match.group(1).strip().rstrip(".")

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) == 1:
        single_line = lines[0].rstrip(".")
        numeric_matches = NUMERIC_TAIL_PATTERN.findall(single_line)
        if numeric_matches:
            return numeric_matches[-1].strip()
        lowered = single_line.lower()
        if lowered in NUMBER_WORDS:
            return single_line

    numeric_matches = NUMERIC_TAIL_PATTERN.findall(cleaned)
    if numeric_matches:
        return numeric_matches[-1].strip()

    return None


def parse_ground_truth_answer(answer_text: str) -> Optional[str]:
    """Extracts the GSM8K final answer after the delimiter."""
    if "####" not in answer_text:
        return None
    _, final_answer = answer_text.split("####", 1)
    return final_answer.strip()


def decode_generated_text(tokenizer: Any, outputs: Any, prompt_length: int) -> str:
    """Decodes only the generated continuation beyond the input prompt."""
    generated_tokens = outputs[:, prompt_length:]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]


def save_evaluation_report(report: Dict[str, Any], report_path: str) -> str:
    """Persists a JSON evaluation report and returns the saved path."""
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    return report_path
