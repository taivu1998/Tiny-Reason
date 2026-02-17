import sys
import os
import re
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import parse_args
from src.utils import setup_logging, seed_everything
from src.dataset import GSM8KProcessor
from src.model import ModelLoader


def normalize_answer(ans: str) -> str:
    """Normalizes answer for flexible matching."""
    ans = ans.strip().lower()
    ans = ans.replace(",", "").replace(" ", "")
    ans = re.sub(r"\.0+$", "", ans)
    ans = re.sub(r"^zero$", "0", ans)
    ans = re.sub(r"^one$", "1", ans)
    ans = re.sub(r"^two$", "2", ans)
    ans = re.sub(r"^three$", "3", ans)
    ans = re.sub(r"^four$", "4", ans)
    ans = re.sub(r"^five$", "5", ans)
    ans = re.sub(r"^six$", "6", ans)
    ans = re.sub(r"^seven$", "7", ans)
    ans = re.sub(r"^eight$", "8", ans)
    ans = re.sub(r"^nine$", "9", ans)
    ans = re.sub(r"^ten$", "10", ans)
    ans = re.sub(r"^\$", "", ans)
    ans = re.sub(r"dollars?$", "", ans)
    ans = re.sub(r"meters?$", "", ans)
    ans = re.sub(r"feet$", "", ans)
    ans = ans.strip()
    return ans


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
    model, tokenizer = ModelLoader.load_for_inference(adapter_path, max_seq_length)

    processor = GSM8KProcessor(config["data"])
    system_prompt = config["data"].get("system_prompt", "You are a helpful assistant.")
    test_data = processor.get_test_set()

    correct_count = 0
    total = len(test_data)

    logger.info(f"Running evaluation on {total} samples...")

    for item in tqdm(test_data):
        prompt = processor._create_prompt(item["question"], system_prompt)

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                do_sample=False,  # Greedy decoding for benchmarking
            )

        response = tokenizer.batch_decode(outputs)[0]

        # Robust Regex: looks for answer, handling potential EOS token variations
        match = re.search(
            r"\*\*Final Answer:\*\*\s*(.*?)(?:<\|im_end\|>|$)", response, re.DOTALL
        )

        if match:
            pred = normalize_answer(match.group(1))
            truth = normalize_answer(item["answer"].split("####")[1].strip())

            if pred == truth:
                correct_count += 1

    accuracy = (correct_count / total) * 100
    logger.info(f"Final Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
