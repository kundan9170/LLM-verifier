import yaml
import re
import numpy as np


class HybridVerifier:
    """
    Production verifier combining four signals:

    1. MATH VERIFICATION  — parses arithmetic expressions from the CoT
       and re-computes them.  If the model writes "68 / 17 = 5", we
       catch it immediately → "incorrect".

    2. COMPLETION DETECTION — regex patterns that fire when the model
       writes a clear conclusion ("the answer is …") or starts
       hallucinating a fake conversation turn ("Human:", "<|im_start|>").

    3. CONFIDENCE TRACKING — rolling mean of per-token logprobs across
       checkpoints.  Sustained high confidence = model is certain.
       Used only as a supporting signal, never triggers exit alone.

    4. REPETITION DETECTION — trigram overlap between the latest chunk
       and recent chunks.  High overlap = the model is stuck in a loop.

    The verifier is STATEFUL: it accumulates history across calls to
    evaluate() within a single question.  Call reset() before each
    new question.

    Returns one of: "correct", "incorrect", "continue".
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ---- Configurable thresholds ----
        v_cfg = self.config.get("verifier", {})
        self.confidence_threshold = v_cfg.get("confidence_threshold", -1.5)
        self.confidence_window = v_cfg.get("confidence_window", 3)
        self.repetition_threshold = v_cfg.get("repetition_threshold", 0.6)
        self.min_checkpoints = v_cfg.get("min_checkpoints_before_correct", 2)

        # ---- Stateful history (reset between questions) ----
        self._reset_state()

        # ---- Conclusion patterns ----
        self.conclusion_patterns = [
            r"the\s+answer\s+is\b",
            r"the\s+final\s+answer\s+is\b",
            r"therefore[,:\s].*?(?:is|=|equals)\s",
            r"thus[,:\s].*?(?:is|=|equals)\s",
            r"in\s+conclusion",
            r"so[,\s]+the\s+(?:answer|result|time|distance|speed|total|value)",
            r"hence[,:\s].*?(?:is|=|equals)\s",
        ]

        # ---- Error / confusion patterns ----
        self.error_patterns = [
            r"this\s+is\s+wrong",
            r"i\s+made\s+a\s+mistake",
            r"wait[,\s]+let\s+me",
            r"that(?:'s|\s+is)\s+incorrect",
            r"contradiction",
            r"doesn(?:'t|t)\s+make\s+sense",
            r"let\s+me\s+reconsider",
            r"actually[,\s]+that(?:'s|\s+is)\s+not\s+right",
        ]

        # ---- Hallucination patterns (fake conversation) ----
        self.hallucination_patterns = [
            r"\bHuman:",
            r"\bUser:",
            r"\bAssistant:",
            r"<\|im_start\|>",
            r"\bQ:",
            r"\bA:",
        ]

    # ==================================================================
    # State management
    # ==================================================================

    def _reset_state(self):
        """Clear all history. Called before each new question."""
        self.checkpoint_count = 0
        self.logprob_history = []       # mean logprob per checkpoint
        self.prev_cot_length = 0        # to extract latest chunk
        self.prev_chunks = []           # for repetition detection
        self.math_results = []          # track verified equations

    def reset(self):
        """Public reset — controller calls this before each run()."""
        self._reset_state()

    # ==================================================================
    # Signal 1: Math verification
    # ==================================================================

    def _math_verify(self, text):
        """
        Find arithmetic expressions like "13 + 4 = 17" and verify them.

        Returns:
            ("incorrect", message)  — if any equation is wrong
            ("verified",  message)  — if equations found and all correct
            ("continue",  None)     — no equations found
        """
        # Pattern: number operator number = number
        # Handles: integers, decimals, negative numbers
        # Operators: + - * / × ÷
        pattern = r'(-?\d+\.?\d*)\s*([+\-*/×÷])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)'

        matches = re.findall(pattern, text)

        if not matches:
            return ("continue", None)

        verified_count = 0

        for left_str, op, right_str, result_str in matches:
            try:
                left = float(left_str)
                right = float(right_str)
                result = float(result_str)

                if op == '+':
                    expected = left + right
                elif op == '-':
                    expected = left - right
                elif op in ('*', '×'):
                    expected = left * right
                elif op in ('/', '÷'):
                    if right == 0:
                        continue
                    expected = left / right
                else:
                    continue

                # Tolerance for floating point
                if abs(expected - result) > 0.01:
                    msg = (
                        f"Arithmetic error: {left_str} {op} {right_str} = {result_str} "
                        f"(expected {expected:.4g})"
                    )
                    return ("incorrect", msg)

                verified_count += 1

            except (ValueError, ZeroDivisionError):
                continue

        if verified_count > 0:
            return ("verified", f"{verified_count} equation(s) verified correct")

        return ("continue", None)

    # ==================================================================
    # Signal 2: Completion detection
    # ==================================================================

    def _check_conclusion(self, text):
        """Check if the model has stated a clear final answer."""
        lower = text.lower()
        for pattern in self.conclusion_patterns:
            if re.search(pattern, lower):
                return True
        return False

    def _check_errors(self, text):
        """Check for self-correction or confusion signals."""
        lower = text.lower()
        for pattern in self.error_patterns:
            if re.search(pattern, lower):
                return True
        return False

    def _check_hallucination(self, text):
        """Check if model started generating fake conversation turns."""
        for pattern in self.hallucination_patterns:
            if re.search(pattern, text):
                return True
        return False

    # ==================================================================
    # Signal 3: Confidence tracking
    # ==================================================================

    def _update_confidence(self, logprobs):
        """
        Track mean logprob per checkpoint.
        Returns True if model has been consistently confident
        over the last N checkpoints.
        """
        if not logprobs:
            return False

        mean_lp = float(np.mean(logprobs))
        self.logprob_history.append(mean_lp)

        if len(self.logprob_history) < self.confidence_window:
            return False

        # Check if last N checkpoints were all above threshold
        recent = self.logprob_history[-self.confidence_window:]
        return all(lp > self.confidence_threshold for lp in recent)

    # ==================================================================
    # Signal 4: Repetition detection
    # ==================================================================

    def _check_repetition(self, current_chunk):
        """
        Compare current chunk against recent chunks using trigram overlap.
        High overlap = model is stuck repeating itself.
        """
        if len(self.prev_chunks) < 2:
            return False

        def get_trigrams(text):
            words = text.lower().split()
            if len(words) < 3:
                return set()
            return set(
                tuple(words[i:i+3])
                for i in range(len(words) - 2)
            )

        current_ngrams = get_trigrams(current_chunk)
        if not current_ngrams:
            return False

        for prev in self.prev_chunks[-3:]:
            prev_ngrams = get_trigrams(prev)
            if not prev_ngrams:
                continue
            overlap = len(current_ngrams & prev_ngrams) / len(current_ngrams)
            if overlap > self.repetition_threshold:
                return True

        return False

    # ==================================================================
    # Main evaluation — combines all signals
    # ==================================================================

    def evaluate(self, full_cot, logprobs=None):
        """
        Inputs:
            full_cot:  the ENTIRE chain-of-thought accumulated so far
            logprobs:  list of per-token logprobs from the LATEST checkpoint

        Returns: "correct", "incorrect", or "continue"

        Decision priority:
            1. Math error found           → "incorrect"  (immediate)
            2. Self-correction detected    → "incorrect"  (immediate)
            3. Hallucination detected      → "correct"    (stop now)
            4. Repetition detected         → "correct"    (model stuck)
            5. Conclusion + confidence     → "correct"    (confident answer)
            6. Conclusion + math verified  → "correct"    (verified answer)
            7. Nothing triggered           → "continue"
        """
        self.checkpoint_count += 1

        # ---- Extract latest chunk for repetition check ----
        current_chunk = full_cot[self.prev_cot_length:]
        self.prev_cot_length = len(full_cot)
        self.prev_chunks.append(current_chunk)

        # ---- Gather all signals ----

        # Signal 1: Math verification
        math_status, math_msg = self._math_verify(full_cot)

        if math_msg:
            print(f"  [Verifier/Math] {math_status}: {math_msg}")

        # Signal 2: Completion / errors / hallucination
        has_conclusion = self._check_conclusion(full_cot)
        has_error = self._check_errors(full_cot)
        has_hallucination = self._check_hallucination(current_chunk)

        # Signal 3: Confidence
        is_confident = self._update_confidence(logprobs)

        # Signal 4: Repetition
        is_repeating = self._check_repetition(current_chunk)

        # ---- Debug logging ----
        print(
            f"  [Verifier] checkpoint={self.checkpoint_count} "
            f"conclusion={has_conclusion} confident={is_confident} "
            f"repeating={is_repeating} math={math_status}"
        )

        # ==============================================================
        # Decision logic (ORDER MATTERS)
        # ==============================================================

        # Priority 1: Arithmetic error → abort immediately
        if math_status == "incorrect":
            return "incorrect"

        # Priority 2: Model is confused / self-correcting → abort
        if has_error:
            return "incorrect"

        # Guard: don't return "correct" until enough reasoning has happened
        if self.checkpoint_count < self.min_checkpoints:
            return "continue"

        # Priority 3: Hallucination → stop before it gets worse
        if has_hallucination:
            print("  [Verifier] Hallucination detected — stopping generation")
            return "correct"

        # Priority 4: Repetition → model is stuck, use what we have
        if is_repeating:
            print("  [Verifier] Repetition detected — stopping generation")
            return "correct"

        # Priority 5: Conclusion stated + sustained confidence
        if has_conclusion and is_confident:
            return "correct"

        # Priority 6: Conclusion stated + math verified
        if has_conclusion and math_status == "verified":
            return "correct"

        # Default: keep generating
        return "continue"
