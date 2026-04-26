import argparse
import yaml

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_engine.model_loader import LLMModelLoader
from pipeline.controller import PipelineController


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # -------------------------------
    # CLI Argument Parser
    # -------------------------------
    parser = argparse.ArgumentParser(description="Run LLM reasoning pipeline.")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Input question for the reasoning pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file."
    )

    args = parser.parse_args()

    # -------------------------------
    # Load config
    # -------------------------------
    config = load_config(args.config)

    print("\n========================================")
    print("      LLM Reasoning Pipeline")
    print("========================================")
    print(f"Model: {config['llm']['model_name']}")
    print("Loading model...")

    # -------------------------------
    # Load LLM (Model + Tokenizer)
    # -------------------------------
    loader = LLMModelLoader(config_path=args.config)
    model, tokenizer = loader.get()

    print("Model loaded successfully!")

    # -------------------------------
    # Instantiate Controller
    # -------------------------------
    controller = PipelineController(model, tokenizer, config_path=args.config)

    # -------------------------------
    # Run Pipeline
    # -------------------------------
    print("\nProcessing question...")
    final_answer = controller.run(args.question)

    print("\n========================================")
    print("                 FINAL ANSWER")
    print("========================================")
    print(final_answer)
    print("========================================\n")


if __name__ == "__main__":
    main()
