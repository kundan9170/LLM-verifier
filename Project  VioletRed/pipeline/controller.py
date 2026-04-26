import yaml
from llm_engine.cot_generator import CoTGenerator
from verifier.llm_verifier import LLMVerifier
from pipeline.early_exit_logic import EarlyExitLogic
from pipeline.utils import print_debug


class PipelineController:
    """
    Qwen3 reasoning pipeline controller with two-phase generation
    and LLM-based verification.

    Phase 1 — THINKING:
      Model generates CoT inside <think>...</think> block.
      After each sentence-aware chunk, the LLM judge (Qwen3-0.6B)
      evaluates whether the step is correct.
      Exits when:
        (a) Model generates </think> naturally, OR
        (b) Verifier says "correct" (conclusion detected) → inject </think>, OR
        (c) Verifier says "incorrect" (bad step) → abort/retry, OR
        (d) EOS hit or max tokens reached.

    Phase 2 — ANSWERING:
      After </think>, model generates the final answer until EOS.
      No verifier checks. Same KV-cache throughout.
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tokenizer = tokenizer

        # Core modules
        self.cot_generator = CoTGenerator(model, tokenizer, config_path)
        self.verifier = LLMVerifier(config_path)
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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful math assistant. Think step by step "
                    "and reason carefully. Put your final answer inside \\boxed{}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{user_query}\n\n"
                    "Please reason step by step, and put your final answer "
                    "within \\boxed{}."
                ),
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
    # Extract answer from CoT if answer phase fails
    # ----------------------------------------------------------------------
    def _extract_answer(self, full_text):
        if "</think>" in full_text:
            answer = full_text.split("</think>")[-1].strip()
            if answer:
                return answer

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
        verifier_call_count = 0
        total_tokens_generated = 0

        # Format prompt with thinking mode
        prompt = self._format_prompt(user_query)

        if self.debug_cfg.get("print_partial_cot"):
            print(f"[Formatted prompt]\n{prompt}\n")

        input_ids = None
        past_key_values = None

        # ==============================================================
        # PHASE 1: THINKING (LLM-verified CoT generation)
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

            # ----- EOS during thinking -----
            if hit_eos:
                print_debug("EOS during thinking phase.", self.debug_cfg)
                answer = self._extract_answer(full_cot)
                if self.debug_cfg["print_final_answer"]:
                    print(f"\n[Final Answer] {answer}")
                return answer , total_tokens_generated

            # ----- LLM Verifier check -----
            verifier_call_count += 1
            verdict = self.verifier.evaluate(
                full_cot,
                logprobs=logprobs,
                question=user_query,   # pass question for judge context
            )

            if self.debug_cfg["print_verifier_decision"]:
                print(f"[Verifier Verdict] {verdict}")

            decision = self.exit_logic.decide(verdict, verifier_call_count)

            print_debug(f"Early exit decision: {decision}", self.debug_cfg)

            # ==============================
            # Handle decisions
            # ==============================
            if decision == "exit":
                print_debug("Verifier exit → injecting </think>", self.debug_cfg)
                input_ids, past_key_values = self.cot_generator.inject_tokens(
                    "</think>\n\n", input_ids, past_key_values
                )
                full_cot += "</think>\n\n"
                think_ended_naturally = True
                break

            elif decision == "abort":
                print_debug("LLM judge detected incorrect reasoning. Aborting.", self.debug_cfg)
                # UPDATE: Return tokens
                return "The reasoning contains an error. Cannot provide a reliable answer.", total_tokens_generated

            else:
                prompt = None
                continue

        # If max tokens reached without </think>
        if not think_ended_naturally:
            print_debug("Max CoT tokens → injecting </think>", self.debug_cfg)
            input_ids, past_key_values = self.cot_generator.inject_tokens(
                "</think>\n\n", input_ids, past_key_values
            )
            full_cot += "</think>\n\n"

        # ==============================================================
        # PHASE 2: ANSWERING (generate until EOS, no verification)
        # ==============================================================
        print_debug("=== Phase 2: Answering ===", self.debug_cfg)

        final_answer, input_ids, past_key_values = (
            self.cot_generator.generate_until_eos(
                input_ids, past_key_values, self.max_answer_tokens
            )
        )
        # UPDATE: Add Phase 2 tokens to the total count
        phase_2_tokens = len(self.tokenizer.encode(final_answer))
        total_tokens_generated += phase_2_tokens

        if self.debug_cfg["print_final_answer"]:
            print(f"\n[Final Answer] {final_answer}")

        if final_answer.strip():
            # UPDATE: Return tuple
            return final_answer.strip(), total_tokens_generated

        print_debug("Answer phase empty — extracting from CoT", self.debug_cfg)
        # UPDATE: Return tuple
        return self._extract_answer(full_cot), total_tokens_generated
