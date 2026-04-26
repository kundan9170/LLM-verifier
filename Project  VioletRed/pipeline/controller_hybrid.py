import yaml
from llm_engine.cot_generator import CoTGenerator
from verifier.hybrid_verifier import HybridVerifier
from pipeline.early_exit_logic import EarlyExitLogic
from pipeline.utils import print_debug


class PipelineController:
    """
    Qwen3 reasoning pipeline controller with two-phase generation.

    Phase 1 — THINKING:
      Model generates CoT inside <think>...</think> block.
      Verifier monitors each sentence-aware chunk.
      Exits when:
        (a) Model generates </think> naturally, OR
        (b) Verifier says "stop" → we inject </think> into the stream, OR
        (c) EOS hit (unusual during thinking), OR
        (d) Max tokens reached.

    Phase 2 — ANSWERING:
      After </think>, model generates the final answer.
      No verifier checks — just generate until EOS.
      Everything stays in the same KV-cache.

    No separate FinalAnswerGenerator needed.
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tokenizer = tokenizer

        # Core modules
        self.cot_generator = CoTGenerator(model, tokenizer, config_path)
        self.verifier = HybridVerifier(config_path)
        self.exit_logic = EarlyExitLogic(config_path)

        # Debug flags
        self.debug_cfg = self.config["debug"]

        # Limits
        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]
        self.max_answer_tokens = self.config["generation"]["max_new_tokens"]

    # ----------------------------------------------------------------------
    # Format prompt with Qwen3 thinking mode
    # ----------------------------------------------------------------------
    def _format_prompt(self, user_query):
        """
        Uses tokenizer.apply_chat_template with enable_thinking=True.

        For Qwen3, this produces:
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant
            <think>

        The model then generates CoT tokens, </think>, and the final answer.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Think step by step and reason carefully."
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
            enable_thinking=True,
        )

        return prompt

    # ----------------------------------------------------------------------
    # Extract final answer from generated text
    # ----------------------------------------------------------------------
    def _extract_answer(self, full_text):
        """
        Extract the part after </think> as the final answer.
        If </think> is not found, return the last meaningful sentence.
        """
        if "</think>" in full_text:
            answer = full_text.split("</think>")[-1].strip()
            if answer:
                return answer

        # Fallback: return last non-empty paragraph
        paragraphs = full_text.strip().split("\n\n")
        for p in reversed(paragraphs):
            if p.strip():
                return p.strip()

        return full_text.strip()

    # ----------------------------------------------------------------------
    # MAIN EXECUTION
    # ----------------------------------------------------------------------
    def run(self, user_query):
        print_debug("Starting pipeline...", self.debug_cfg)

        # Reset verifier state
        self.verifier.reset()

        full_cot = ""
        full_answer = ""
        verifier_call_count = 0
        total_tokens_generated = 0

        # Format prompt with thinking mode
        prompt = self._format_prompt(user_query)

        if self.debug_cfg.get("print_partial_cot"):
            print(f"[Formatted prompt]\n{prompt}\n")

        input_ids = None
        past_key_values = None

        # ==============================================================
        # PHASE 1: THINKING (verified CoT generation)
        # ==============================================================
        print_debug("=== Phase 1: Thinking ===", self.debug_cfg)

        think_ended_naturally = False

        while total_tokens_generated < self.max_cot_tokens:

            source = prompt if input_ids is None else input_ids

            (partial_text, logprobs, input_ids, past_key_values,
             hit_eos, hit_think_end) = (
                self.cot_generator.generate_until_checkpoint(source, past_key_values)
            )

            full_cot += partial_text
            total_tokens_generated += len(logprobs) if logprobs else len(partial_text.split())

            if self.debug_cfg["print_partial_cot"]:
                print(f"\n[Think chunk {verifier_call_count + 1} | ~{total_tokens_generated} tokens]")
                print(f"  {partial_text.strip()}")

            # ----- Model generated </think> naturally -----
            if hit_think_end:
                print_debug("Model generated </think> naturally.", self.debug_cfg)
                think_ended_naturally = True
                break

            # ----- EOS during thinking (unusual) -----
            if hit_eos:
                print_debug("EOS during thinking phase. Extracting answer from CoT.", self.debug_cfg)
                answer = self._extract_answer(full_cot)
                if self.debug_cfg["print_final_answer"]:
                    print(f"\n[Final Answer] {answer}")
                return answer

            # ----- Verifier check -----
            verifier_call_count += 1
            verdict = self.verifier.evaluate(full_cot, logprobs)

            if self.debug_cfg["print_verifier_decision"]:
                print(f"[Verifier] {verdict}")

            decision = self.exit_logic.decide(verdict, verifier_call_count)

            if decision == "exit":
                # Inject </think> to force transition to answering
                print_debug("Verifier exit → injecting </think>", self.debug_cfg)

                input_ids, past_key_values = self.cot_generator.inject_tokens(
                    "</think>\n\n", input_ids, past_key_values
                )
                full_cot += "</think>\n\n"
                think_ended_naturally = True
                break

            elif decision == "abort":
                print_debug("Verifier abort — reasoning incorrect.", self.debug_cfg)
                return "The reasoning seems incorrect. Cannot provide a reliable answer."

            else:
                # Continue thinking
                prompt = None
                continue

        # If max tokens reached without </think>, inject it
        if not think_ended_naturally:
            print_debug("Max CoT tokens → injecting </think>", self.debug_cfg)
            input_ids, past_key_values = self.cot_generator.inject_tokens(
                "</think>\n\n", input_ids, past_key_values
            )
            full_cot += "</think>\n\n"

        # ==============================================================
        # PHASE 2: ANSWERING (unverified, generate until EOS)
        # ==============================================================
        print_debug("=== Phase 2: Answering ===", self.debug_cfg)

        final_answer, input_ids, past_key_values = (
            self.cot_generator.generate_until_eos(
                input_ids, past_key_values, self.max_answer_tokens
            )
        )

        if self.debug_cfg["print_final_answer"]:
            print(f"\n[Final Answer] {final_answer}")

        # If model produced something, use it directly
        if final_answer.strip():
            return final_answer.strip()

        # Fallback: extract from CoT if answer phase produced nothing
        print_debug("Answer phase empty — extracting from CoT", self.debug_cfg)
        return self._extract_answer(full_cot)