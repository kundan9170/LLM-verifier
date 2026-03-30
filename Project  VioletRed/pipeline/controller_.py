import yaml
from llm_engine.cot_generator import CoTGenerator
from llm_engine.final_answer_generator import FinalAnswerGenerator
from verifier.hybrid_verifier import HybridVerifier
from pipeline.early_exit_logic import EarlyExitLogic
from pipeline.utils import print_debug


class PipelineController:
    """
    Full reasoning pipeline controller.

    Workflow:
      1. Format prompt using the model's chat template.
      2. Generate CoT in sentence-aware chunks.
      3. After each complete sentence → send to verifier.
      4. Verifier returns: correct / incorrect / continue.
      5. EarlyExitLogic decides: exit / continue / abort.
      6. If EOS token detected → generate final answer immediately.
      7. If exit → generate final concise answer.
      8. If continue → keep generating CoT.
      9. If abort → output fallback.
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tokenizer = tokenizer

        # Core modules
        self.cot_generator = CoTGenerator(model, tokenizer, config_path)
        self.final_answer_gen = FinalAnswerGenerator(model, tokenizer, config_path)
        self.verifier = HybridVerifier(config_path)
        self.exit_logic = EarlyExitLogic(config_path)

        # Debug flags
        self.debug_cfg = self.config["debug"]

        # Hard cap on total reasoning tokens
        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]

    # ----------------------------------------------------------------------
    # Format prompt using the model's chat template
    # ----------------------------------------------------------------------
    def _format_prompt(self, user_query):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Think step by step, reason carefully, "
                    "and provide a clear final answer."
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt

    # ----------------------------------------------------------------------
    # MAIN EXECUTION ENTRY POINT
    # ----------------------------------------------------------------------
    def run(self, user_query):
        print_debug("Starting pipeline...", self.debug_cfg)

        # Reset verifier state for this new question
        self.verifier.reset()

        full_cot = ""
        verifier_call_count = 0
        total_tokens_generated = 0

        prompt = self._format_prompt(user_query)

        if self.debug_cfg.get("print_partial_cot"):
            print(f"[DEBUG] Formatted prompt:\n{prompt}\n")

        input_ids = None
        past_key_values = None

        # ======================================================
        # MAIN GENERATION LOOP
        # Runs until: EOS, early exit, abort, or max tokens
        # Each iteration generates one sentence-aware chunk
        # ======================================================
        while total_tokens_generated < self.max_cot_tokens:

            source = prompt if input_ids is None else input_ids

            partial_text, logprobs, input_ids, past_key_values, hit_eos = (
                self.cot_generator.generate_until_checkpoint(source, past_key_values)
            )

            full_cot += partial_text
            total_tokens_generated += len(logprobs) if logprobs else len(partial_text.split())

            if self.debug_cfg["print_partial_cot"]:
                print(f"\n[Chunk {verifier_call_count + 1} | ~{total_tokens_generated} tokens]")
                print(f"  {partial_text.strip()}")

            # ----- EOS: model finished its response naturally -----
            if hit_eos:
                print_debug("EOS token detected. Generating final answer...", self.debug_cfg)
                final_answer = self.final_answer_gen.generate_final_answer(full_cot)
                if self.debug_cfg["print_final_answer"]:
                    print(f"\n[Final Answer] {final_answer}")
                return final_answer

            # ----- Verifier check -----
            verifier_call_count += 1
            verdict = self.verifier.evaluate(full_cot, logprobs)

            if self.debug_cfg["print_verifier_decision"]:
                print(f"[Verifier Verdict] {verdict}")

            decision = self.exit_logic.decide(verdict, verifier_call_count)

            print_debug(f"Early exit decision: {decision}", self.debug_cfg)

            # ==============================
            # Handle controller decisions
            # ==============================
            if decision == "exit":
                print_debug("Early exit triggered. Generating final answer...", self.debug_cfg)
                final_answer = self.final_answer_gen.generate_final_answer(full_cot)
                if self.debug_cfg["print_final_answer"]:
                    print(f"\n[Final Answer] {final_answer}")
                return final_answer

            elif decision == "abort":
                print_debug("Verifier marked reasoning as incorrect. Aborting.", self.debug_cfg)
                return "The reasoning seems incorrect. Cannot provide a reliable answer."

            else:
                prompt = None
                continue

        # ======================================================
        # Max tokens reached
        # ======================================================
        print_debug("Max CoT reached. Generating final answer...", self.debug_cfg)
        final_answer = self.final_answer_gen.generate_final_answer(full_cot)
        return final_answer