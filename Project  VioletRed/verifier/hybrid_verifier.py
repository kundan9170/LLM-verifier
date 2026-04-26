import yaml
import re
import numpy as np


class HybridVerifier:
    """
    Production verifier combining four signals:

    1. MATH VERIFICATION  — parses arithmetic expressions from the CoT
       and re-computes them.

    2. COMPLETION DETECTION — regex patterns for conclusions, errors,
       and hallucinated conversation turns.

    3. CONFIDENCE TRACKING — rolling mean of per-token logprobs.
       Supporting signal only, never triggers exit alone.

    4. REPETITION DETECTION — trigram overlap on NATURAL LANGUAGE words
       only (math symbols stripped). Requires overlap with 2+ previous
       chunks to confirm a genuine loop.

    The verifier is STATEFUL. Call reset() before each new question.
    Returns: "correct", "incorrect", or "continue".
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        v_cfg = self.config.get("verifier", {})
        self.confidence_threshold = v_cfg.get("confidence_threshold", -1.5)
        self.confidence_window = v_cfg.get("confidence_window", 3)
        self.repetition_threshold = v_cfg.get("repetition_threshold", 0.6)
        self.min_checkpoints = v_cfg.get("min_checkpoints_before_correct", 3)

        self._reset_state()

        # ---- Conclusion patterns ----
        self.conclusion_patterns = [
            r"the\s+answer\s+is\b",
            r"the\s+final\s+answer\s+is\b",
            r"therefore[,:\s].*?(?:is|=|equals)\s",
            r"thus[,:\s].*?(?:is|=|equals)\s",
            r"in\s+conclusion",
            r"so[,\s]+the\s+(?:answer|result|time|distance|speed|total|value|son|father|age|prime)",
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

        # ---- Hallucination patterns ----
        self.hallucination_patterns = [
            r"\bHuman:",
            r"\bUser:",
            r"\bAssistant:",
            r"<\|im_start\|>",
        ]

        # ---- Symbols to strip before repetition check ----
        # Matches LaTeX commands, math operators, brackets, single letters
        # that are likely variable names, and digits
        self.math_strip_pattern = re.compile(
            r'\\[a-zA-Z]+|'       # LaTeX commands: \frac, \sqrt, etc.
            r'[+\-*/=^(){}\[\]\\|<>×÷]|'  # math operators and brackets
            r'\b\d+\.?\d*\b|'     # numbers
            r'\b[a-zA-Z]\b|'      # single-letter variables: n, p, x
            r'\$+'                # dollar signs (LaTeX delimiters)
        )

    # ==================================================================
    # State management
    # ==================================================================

    def _reset_state(self):
        self.checkpoint_count = 0
        self.logprob_history = []
        self.prev_cot_length = 0
        self.prev_chunks = []
        self.math_results = []

    def reset(self):
        self._reset_state()

    # ==================================================================
    # Signal 1: Math verification
    # ==================================================================

    def _math_verify(self, text):
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
        lower = text.lower()
        for pattern in self.conclusion_patterns:
            if re.search(pattern, lower):
                return True
        return False

    def _check_errors(self, text):
        lower = text.lower()
        for pattern in self.error_patterns:
            if re.search(pattern, lower):
                return True
        return False

    def _check_hallucination(self, text):
        for pattern in self.hallucination_patterns:
            if re.search(pattern, text):
                return True
        return False

    # ==================================================================
    # Signal 3: Confidence tracking
    # ==================================================================

    def _update_confidence(self, logprobs):
        if not logprobs:
            return False

        mean_lp = float(np.mean(logprobs))
        self.logprob_history.append(mean_lp)

        if len(self.logprob_history) < self.confidence_window:
            return False

        recent = self.logprob_history[-self.confidence_window:]
        return all(lp > self.confidence_threshold for lp in recent)

    # ==================================================================
    # Signal 4: Repetition detection (math-aware)
    # ==================================================================

    def _strip_math(self, text):
        """
        Remove math symbols, LaTeX, numbers, and single-letter variables.
        Keeps only natural language words so that trigram comparison
        measures semantic repetition, not algebraic symbol reuse.

        Example:
          "16p = (n - 1)(n^2 + n + 1)"  →  ""
          "We can factor using the difference of cubes formula"
              → "we can factor using the difference of cubes formula"
        """
        stripped = self.math_strip_pattern.sub(' ', text)
        # Collapse whitespace and lowercase
        stripped = re.sub(r'\s+', ' ', stripped).strip().lower()
        return stripped

    def _get_trigrams(self, text):
        """Extract trigrams from natural language text (math stripped)."""
        cleaned = self._strip_math(text)
        words = cleaned.split()
        if len(words) < 3:
            return set()
        return set(
            tuple(words[i:i+3])
            for i in range(len(words) - 2)
        )

    def _check_repetition(self, current_chunk):
        """
        Compare current chunk against PREVIOUS chunks using trigram overlap
        on natural language words only (math symbols stripped).

        To confirm a real loop, we require overlap with at least 2
        previous chunks — a single overlap could just be the model
        referring back to a previous step.

        IMPORTANT: current_chunk must NOT be in self.prev_chunks yet.
        """
        # Need at least 3 previous chunks to detect a pattern
        if len(self.prev_chunks) < 3:
            return False

        current_ngrams = self._get_trigrams(current_chunk)
        if not current_ngrams or len(current_ngrams) < 3:
            return False

        overlap_count = 0

        for prev in self.prev_chunks[-4:]:
            prev_ngrams = self._get_trigrams(prev)
            if not prev_ngrams:
                continue
            overlap = len(current_ngrams & prev_ngrams) / len(current_ngrams)
            if overlap > self.repetition_threshold:
                overlap_count += 1

        # Only flag if 2+ previous chunks are highly similar
        return overlap_count >= 2

    # ==================================================================
    # Main evaluation
    # ==================================================================

    def evaluate(self, full_cot, logprobs=None):
        """
        Returns: "correct", "incorrect", or "continue"

        Decision priority:
            1. Math error found           → "incorrect"  (immediate)
            2. Self-correction detected    → "incorrect"  (immediate)
               GUARD: checkpoint <= min    → "continue"   (too early)
            3. Hallucination detected      → "correct"    (stop now)
            4. Repetition detected         → "correct"    (model stuck)
            5. Conclusion + confidence     → "correct"    (confident answer)
            6. Conclusion + math verified  → "correct"    (verified answer)
            7. Nothing triggered           → "continue"
        """
        self.checkpoint_count += 1

        # ---- Extract latest chunk ----
        current_chunk = full_cot[self.prev_cot_length:]
        self.prev_cot_length = len(full_cot)

        # ---- Gather all signals ----

        math_status, math_msg = self._math_verify(full_cot)
        if math_msg:
            print(f"  [Verifier/Math] {math_status}: {math_msg}")

        has_conclusion = self._check_conclusion(full_cot)
        has_error = self._check_errors(full_cot)
        has_hallucination = self._check_hallucination(current_chunk)

        is_confident = self._update_confidence(logprobs)

        # Repetition — check BEFORE appending
        is_repeating = self._check_repetition(current_chunk)

        # NOW append to history
        self.prev_chunks.append(current_chunk)

        # ---- Debug ----
        print(
            f"  [Verifier] checkpoint={self.checkpoint_count} "
            f"conclusion={has_conclusion} confident={is_confident} "
            f"repeating={is_repeating} math={math_status}"
        )

        # ==============================================================
        # Decision logic
        # ==============================================================

        if math_status == "incorrect":
            return "incorrect"

        if has_error:
            return "incorrect"

        if self.checkpoint_count <= self.min_checkpoints:
            return "continue"

        if has_hallucination:
            print("  [Verifier] Hallucination detected — stopping generation")
            return "correct"

        if is_repeating:
            print("  [Verifier] Repetition detected — stopping generation")
            return "correct"

        if has_conclusion and is_confident:
            return "correct"

        if has_conclusion and math_status == "verified":
            return "correct"

        return "continue"