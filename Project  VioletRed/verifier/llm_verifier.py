import yaml
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMVerifier:
    """
    Verifier that uses a small separate LLM (Qwen3-0.6B) as a judge
    to detect wrong reasoning steps.

    Two responsibilities:
      1. ERROR DETECTION (LLM judge) — after each chunk, asks the 0.6B
         model: "Is this reasoning step correct?" If the judge says no,
         returns "incorrect" so the pipeline can abort/retry.

      2. COMPLETION DETECTION (lightweight text check) — detects when
         the model has stated a final answer ("the answer is ...") or
         started hallucinating. Returns "correct" meaning "done, exit."

    The LLM judge is the core innovation — it catches semantic errors
    that no regex can detect (e.g., "since p is prime, p must be even").

    The verifier is STATEFUL. Call reset() before each new question.
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        v_cfg = self.config.get("verifier", {})
        self.judge_model_name = v_cfg.get("judge_model", "Qwen/Qwen3-0.6B")
        self.judge_max_tokens = v_cfg.get("judge_max_tokens", 64)
        self.min_checkpoints = v_cfg.get("min_checkpoints_before_correct", 3)
        self.error_threshold = v_cfg.get("error_confidence_threshold", 0.6)

        # Load judge model
        self._load_judge()

        # State
        self._reset_state()

        # Conclusion patterns (lightweight, no LLM needed)
        self.conclusion_patterns = [
            r"the\s+answer\s+is\b",
            r"the\s+final\s+answer\s+is\b",
            r"therefore[,:\s].*?(?:is|=|equals)\s",
            r"thus[,:\s].*?(?:is|=|equals)\s",
            r"in\s+conclusion",
            r"so[,\s]+the\s+(?:answer|result|time|distance|speed|total|value|prime|son|father|age)",
            r"hence[,:\s].*?(?:is|=|equals)\s",
            r"\\boxed\{",
        ]

        # Hallucination patterns
        self.hallucination_patterns = [
            r"\bHuman:",
            r"\bUser:",
            r"\bAssistant:",
            r"<\|im_start\|>",
        ]

    # ==================================================================
    # Load the judge model (separate from the generator)
    # ==================================================================

    # def _load_judge(self):
    #     """Load Qwen3-0.6B as the verification judge."""
    #     print(f"[LLMVerifier] Loading judge model: {self.judge_model_name}")

    #     device = self.config.get("verifier", {}).get("judge_device", "auto")
    #     dtype_str = self.config.get("verifier", {}).get("judge_dtype", "float16")

    #     if dtype_str == "float16":
    #         dtype = torch.float16
    #     elif dtype_str == "bfloat16":
    #         dtype = torch.bfloat16
    #     else:
    #         dtype = torch.float32

    #     self.judge_tokenizer = AutoTokenizer.from_pretrained(
    #         self.judge_model_name, use_fast=True
    #     )

    #     self.judge_model = AutoModelForCausalLM.from_pretrained(
    #         self.judge_model_name,
    #         torch_dtype=dtype,
    #         device_map=device,
    #     )
    #     self.judge_model.eval()

    #     print(f"[LLMVerifier] Judge model loaded successfully.")
    def _load_judge(self):
        """Load Qwen3-0.6B as the verification judge."""
        print(f"[LLMVerifier] Loading judge model: {self.judge_model_name}")

        device = self.config.get("verifier", {}).get("judge_device", "auto")
        dtype_str = self.config.get("verifier", {}).get("judge_dtype", "float16")

        if dtype_str == "float16":
            dtype = torch.float16
        elif dtype_str == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.judge_tokenizer = AutoTokenizer.from_pretrained(
            self.judge_model_name, use_fast=True
        )

        self.judge_model = AutoModelForCausalLM.from_pretrained(
            self.judge_model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="sdpa" # OPTIMIZATION: Flash Attention for the judge
        )
        self.judge_model.eval()
        
        # # OPTIMIZATION: Compile the judge model for faster repeated inference calls
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     self.judge_model = torch.compile(self.judge_model, mode="reduce-overhead")

        print(f"[LLMVerifier] Judge model loaded successfully.")

    # ==================================================================
    # State management
    # ==================================================================

    def _reset_state(self):
        self.checkpoint_count = 0
        self.prev_cot_length = 0
        self.steps = []           # list of individual step strings
        self.step_scores = []     # judge verdict per step

    def reset(self):
        self._reset_state()

    # ==================================================================
    # LLM Judge — core error detection
    # ==================================================================

    def _judge_step(self, question, steps_so_far, current_step):
        """
        Ask the 0.6B judge model whether the current reasoning step
        is correct.

        Returns:
            ("correct", confidence) or ("wrong", confidence)
        """

        # Build context: show previous steps as numbered list
        if steps_so_far:
            prev_text = "\n".join(
                f"Step {i+1}: {s.strip()}"
                for i, s in enumerate(steps_so_far)
            )
        else:
            prev_text = "(No previous steps)"

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a math reasoning verifier. "
                    "Your job is to check if a reasoning step is mathematically "
                    "correct and logically follows from previous steps. "
                    "Reply with ONLY 'correct' or 'wrong' followed by "
                    "a brief reason (one sentence max)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Previous steps:\n{prev_text}\n\n"
                    f"Current step to verify:\n{current_step.strip()}\n\n"
                    f"Is this step correct or wrong?"
                ),
            },
        ]

        prompt = self.judge_tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # non-thinking mode for speed
        )

        inputs = self.judge_tokenizer(
            prompt, return_tensors="pt"
        ).to(self.judge_model.device)

        with torch.no_grad():
            output_ids = self.judge_model.generate(
                **inputs,
                max_new_tokens=self.judge_max_tokens,
                temperature=0.3,     # low temp for deterministic judgment
                top_p=0.9,
                do_sample=True,
            )

        response = self.judge_tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        print(f"  [Judge raw] {response[:100]}")

        # Parse the verdict
        if "wrong" in response[:30] or "incorrect" in response[:30] or "error" in response[:30]:
            return ("wrong", 0.8)
        elif "correct" in response[:30] or "right" in response[:30] or "valid" in response[:30]:
            return ("correct", 0.8)
        else:
            # Ambiguous — default to correct (don't block on uncertainty)
            return ("correct", 0.3)

    # ==================================================================
    # Completion detection (lightweight, no LLM)
    # ==================================================================

    def _check_conclusion(self, text):
        lower = text.lower()
        for pattern in self.conclusion_patterns:
            if re.search(pattern, lower):
                return True
        return False

    def _check_hallucination(self, text):
        for pattern in self.hallucination_patterns:
            if re.search(pattern, text):
                return True
        return False

    # ==================================================================
    # Main evaluation
    # ==================================================================

    def evaluate(self, full_cot, logprobs=None, question=""):
        """
        Inputs:
            full_cot:  entire chain-of-thought so far
            logprobs:  per-token logprobs (unused, kept for API compat)
            question:  the original user question (needed for judge context)

        Returns: "correct", "incorrect", or "continue"

        Logic:
            1. Extract latest chunk
            2. Ask LLM judge if the step is correct
            3. If judge says wrong → "incorrect" (abort)
            4. If hallucination detected → "correct" (stop, extract)
            5. If conclusion detected + enough steps → "correct" (done)
            6. Otherwise → "continue"
        """
        self.checkpoint_count += 1

        # Extract latest chunk
        current_chunk = full_cot[self.prev_cot_length:]
        self.prev_cot_length = len(full_cot)

        # Skip near-empty chunks
        if len(current_chunk.strip()) < 5:
            print(f"  [Verifier] checkpoint={self.checkpoint_count} — skipping tiny chunk")
            return "continue"

        # ------ LLM Judge: is this step correct? ------
        verdict, confidence = self._judge_step(
            question=question,
            steps_so_far=self.steps,
            current_step=current_chunk,
        )

        self.steps.append(current_chunk)
        self.step_scores.append(verdict)

        # ------ Lightweight checks ------
        has_conclusion = self._check_conclusion(full_cot)
        has_hallucination = self._check_hallucination(current_chunk)

        # ------ Debug ------
        print(
            f"  [Verifier] checkpoint={self.checkpoint_count} "
            f"judge={verdict}({confidence:.1f}) "
            f"conclusion={has_conclusion} "
            f"hallucination={has_hallucination}"
        )

        # ==============================================================
        # Decision logic
        # ==============================================================

        # Priority 1: Judge says step is WRONG with high confidence
        if verdict == "wrong" and confidence >= self.error_threshold:
            print(f"  [Verifier] Judge detected error in step {self.checkpoint_count}")
            return "incorrect"

        # Guard: don't return "correct" too early
        if self.checkpoint_count <= self.min_checkpoints:
            return "continue"

        # Priority 2: Hallucination → stop and extract
        if has_hallucination:
            print("  [Verifier] Hallucination detected — stopping")
            return "correct"

        # Priority 3: Conclusion detected → reasoning is done
        if has_conclusion:
            return "correct"

        # Default: keep generating
        return "continue"