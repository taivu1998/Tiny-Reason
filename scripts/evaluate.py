import sys
import os
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import parse_args
from src.utils import setup_logging, seed_everything
from src.dataset import GSM8KProcessor
from src.model import ModelLoader
from src.runtime import resolve_device, ensure_supported_4bit_runtime
from src.evaluation_utils import (
    decode_generated_text,
    extract_final_answer,
    normalize_answer,
    parse_ground_truth_answer,
    save_evaluation_report,
)


def main():
    config = parse_args()
    logger = setup_logging()
    seed_everything(config["project"]["seed"])

    adapter_path = config["project"].get("adapter_path", "outputs/lora_adapter")

    if not os.path.exists(adapter_path):
        logger.error(f"Adapter not found at {adapter_path}. Run training first.")
        return

    logger.info(f"Loading adapter from {adapter_path}...")
    max_seq_length = config["model"].get("max_seq_length", 2048)
    device = resolve_device()
    load_in_4bit = config["model"].get("load_in_4bit", True)
    ensure_supported_4bit_runtime(device, load_in_4bit)
    model, tokenizer = ModelLoader.load_for_inference(
        adapter_path,
        max_seq_length,
        load_in_4bit=load_in_4bit,
        device=device,
    )

    processor = GSM8KProcessor(config["data"])
    system_prompt = config["data"].get("system_prompt", "You are a helpful assistant.")
    test_data = processor.get_test_set()

    correct_count = 0
    normalized_match_count = 0
    unparsable_predictions = 0
    malformed_ground_truth = 0
    total = len(test_data)
    prediction_rows = []

    logger.info(f"Running evaluation on {total} samples with device={device}...")

    for index, item in enumerate(tqdm(test_data)):
        prompt = processor.create_prompt(item["question"], system_prompt)

        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                do_sample=False,  # Greedy decoding for benchmarking
            )

        generated_text = decode_generated_text(tokenizer, outputs, prompt_length)
        predicted_answer = extract_final_answer(generated_text)
        truth_answer = parse_ground_truth_answer(item["answer"])

        normalized_prediction = (
            normalize_answer(predicted_answer) if predicted_answer is not None else None
        )
        normalized_truth = (
            normalize_answer(truth_answer) if truth_answer is not None else None
        )
        exact_match = predicted_answer == truth_answer and truth_answer is not None
        normalized_match = (
            normalized_prediction == normalized_truth
            if normalized_prediction is not None and normalized_truth is not None
            else False
        )

        if predicted_answer is None:
            unparsable_predictions += 1
        if truth_answer is None:
            malformed_ground_truth += 1

        if exact_match:
            correct_count += 1
        if normalized_match:
            normalized_match_count += 1

        prediction_rows.append(
            {
                "index": index,
                "question": item["question"],
                "prediction": predicted_answer,
                "ground_truth": truth_answer,
                "normalized_prediction": normalized_prediction,
                "normalized_ground_truth": normalized_truth,
                "exact_match": exact_match,
                "normalized_match": normalized_match,
            }
        )

    accuracy = (correct_count / total) * 100 if total else 0.0
    normalized_accuracy = (normalized_match_count / total) * 100 if total else 0.0
    report_path = save_evaluation_report(
        {
            "adapter_path": adapter_path,
            "device": str(device),
            "total_samples": total,
            "exact_match_accuracy": accuracy,
            "normalized_match_accuracy": normalized_accuracy,
            "exact_match_count": correct_count,
            "normalized_match_count": normalized_match_count,
            "unparsable_predictions": unparsable_predictions,
            "malformed_ground_truth": malformed_ground_truth,
            "predictions": prediction_rows,
        },
        os.path.join("results", "evaluation_report.json"),
    )
    logger.info(f"Final Exact-Match Accuracy: {accuracy:.2f}%")
    logger.info(f"Final Normalized Accuracy: {normalized_accuracy:.2f}%")
    logger.info(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
