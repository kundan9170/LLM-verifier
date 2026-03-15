import yaml
from llm_engine.cot_generator import CoTGenerator
from llm_engine.final_answer_generator import FinalAnswerGenerator
from verifier.rule_based_verifier import RuleBasedVerifier
from pipeline.early_exit_logic import EarlyExitLogic
from pipeline.utils import print_debug


class PipelineController:
    """
    Full reasoning pipeline controller.

    Workflow:
      1. Start generating chain-of-thought (CoT).
      2. Every N tokens → send partial CoT to verifier.
      3. Verifier returns: correct / incorrect / continue.
      4. EarlyExitLogic decides: exit / continue / abort.
      5. If exit → generate final concise answer.
      6. If continue → keep generating CoT.
      7. If abort → output fallback or request refinement.
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Core modules
        self.cot_generator = CoTGenerator(model, tokenizer, config_path)
        self.final_answer_gen = FinalAnswerGenerator(model, tokenizer, config_path)
        self.verifier = RuleBasedVerifier(config_path)
        self.exit_logic = EarlyExitLogic(config_path)

        # Debug flags
        self.debug_cfg = self.config["debug"]

        # LLM generation settings
        self.checkpoint_interval = self.config["llm"]["checkpoint_interval"]
        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]

    # ----------------------------------------------------------------------
    # MAIN EXECUTION ENTRY POINT
    # ----------------------------------------------------------------------
    def run(self, user_query):
        """
        Executes the full pipeline and returns the final answer.
        """

        print_debug("Starting pipeline...", self.debug_cfg)

        # The CoT string we build up gradually
        full_cot = ""
        # Count how many times verifier was called
        verifier_call_count = 0

        # Initial prompt that instructs model to think step-by-step
        prompt = (
            f"{user_query}\n"
            # f"Think step by step and reason carefully.\n"
        )

        # Prepare for incremental generation
        input_ids = None  # will be updated per checkpoint

        # ======================================================
        # MAIN GENERATION LOOP
        # ======================================================
        for global_step in range(0, self.max_cot_tokens, self.checkpoint_interval):

            # 1. Generate until next checkpoint
            # First iteration: input_ids is None → pass prompt
            # Later iterations: pass input_ids tensor
            source = prompt if input_ids is None else input_ids

            partial_text, logprobs, input_ids = self.cot_generator.generate_until_checkpoint(source)


            full_cot += partial_text

            if self.debug_cfg["print_partial_cot"]:
                print("\n[DEBUG Partial CoT]")
                print(partial_text)

            # 2. Call verifier
            verifier_call_count += 1
            verdict = self.verifier.evaluate(full_cot, logprobs)

            if self.debug_cfg["print_verifier_decision"]:
                print(f"[Verifier Verdict] {verdict}")

            # 3. Decide what to do
            decision = self.exit_logic.decide(verdict, verifier_call_count)

            # Add debug logs to trace intermediate reasoning and decisions
            print("[DEBUG] Partial CoT reasoning:", partial_text)
            print("[DEBUG] Verifier verdict:", verdict)
            print("[DEBUG] Early exit decision:", decision)

            # ==============================
            # Handle controller decisions
            # ==============================
            if decision == "exit":
                print_debug("Early exit triggered. Generating final answer...", self.debug_cfg)

                final_answer = self.final_answer_gen.generate_final_answer(full_cot)

                if self.debug_cfg["print_final_answer"]:
                    print(f"[Final Answer] {final_answer}")

                return final_answer

            elif decision == "abort":
                print_debug("Verifier marked reasoning as incorrect. Aborting pipeline.", self.debug_cfg)
                return "The reasoning seems incorrect. Cannot provide a reliable answer."

            else:
                # Continue generating reasoning
                prompt = None
                continue

        # ======================================================
        # If we reach here → max tokens reached, no early exit
        # ======================================================
        print_debug("Max CoT reached. Generating final answer...", self.debug_cfg)
        final_answer = self.final_answer_gen.generate_final_answer(full_cot)

        return final_answer
