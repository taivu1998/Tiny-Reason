import sys
import os
import torch
from transformers import TextStreamer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import parse_args
from src.model import ModelLoader
from src.dataset import GSM8KProcessor


def main():
    config = parse_args()

    adapter_path = config["project"].get("adapter_path", "outputs/lora_adapter")

    if not os.path.exists(adapter_path):
        print(f"Error: Adapter not found at {adapter_path}")
        print("Please run training first: make train")
        return

    print(f"Loading model from {adapter_path}...")
    max_seq_length = config["model"].get("max_seq_length", 2048)
    model, tokenizer = ModelLoader.load_for_inference(adapter_path, max_seq_length)

    processor = GSM8KProcessor(config["data"])
    system_prompt = config["data"].get("system_prompt", "You are a helpful assistant.")

    print("\n==========================================")
    print("TinyReason Interactive Demo (Type 'exit' to quit)")
    print("==========================================\n")

    while True:
        question = input("Problem: ")
        if question.lower() in ["exit", "quit"]:
            break

        prompt = processor._create_prompt(question, system_prompt)

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Streamer allows us to see the CoT generation in real-time
        streamer = TextStreamer(tokenizer, skip_prompt=True)

        print("\nReasoning...")
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            use_cache=True,
            do_sample=True,
            temperature=0.1,
        )
        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
