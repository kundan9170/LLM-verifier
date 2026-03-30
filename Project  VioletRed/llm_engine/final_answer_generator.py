import yaml
import re
import torch


class FinalAnswerGenerator:
    """
    Converts partial chain-of-thought into a final concise answer.
    Uses:
      - LLM summarisation (primary)
      - Rule-based extraction (fallback)
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        self.model = model
        self.tokenizer = tokenizer

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.temperature = self.config["generation"]["temperature"]
        self.top_p = self.config["generation"]["top_p"]
        self.max_new_tokens = self.config["generation"]["max_new_tokens"]

    def _encode(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def generate_with_llm(self, partial_cot):
        prompt = (
            "You have been reasoning about a question. "
            "Based on the reasoning below, provide ONLY the final concise answer.\n"
            "Do not explain your steps.\n\n"
            f"Reasoning so far:\n{partial_cot}\n\n"
            "Final Answer:"
        )

        inputs = self._encode(prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

        output_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return output_text.strip()

    def heuristic_extract(self, partial_cot):
        """
        Smarter heuristic: looks for conclusion sentences first,
        falls back to last number (for math), then last sentence.
        """

        # Look for sentences with conclusion markers
        sentences = re.split(r'[.\n]', partial_cot)
        conclusion_markers = [
            "answer is", "result is", "capital is", "therefore",
            "thus", "in conclusion", "final answer",
        ]

        for sent in reversed(sentences):
            lower = sent.lower().strip()
            if any(kw in lower for kw in conclusion_markers):
                # Try to extract the part after "is"
                match = re.search(r'\bis\b\s+(.+)', sent, re.IGNORECASE)
                if match:
                    return match.group(1).strip().rstrip('.')
                return sent.strip()

        # Fallback: last number (for math questions)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", partial_cot)
        if numbers:
            return numbers[-1]

        # Fallback: last non-empty sentence
        for sent in reversed(sentences):
            if sent.strip():
                return sent.strip()

        return partial_cot.strip()

    def generate_final_answer(self, partial_cot):
        """
        First tries LLM summarisation.
        If it fails (empty output), uses heuristic fallback.
        """

        try:
            answer = self.generate_with_llm(partial_cot)
            if answer:
                return answer
        except Exception as e:
            print(f"[FinalAnswerGenerator] LLM summarisation failed: {e}")

        # fallback
        return self.heuristic_extract(partial_cot)
