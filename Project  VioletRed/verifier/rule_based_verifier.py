import yaml
import numpy as np


class RuleBasedVerifier:
    """
    A simple pluggable rule-based verifier.

    It returns one of:
      - "continue"
      - "correct"
      - "incorrect"

    It uses lightweight heuristics:
      (A) Keyword-based signals
      (B) Logprob stability heuristic
      (C) Pattern / structure heuristic
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Used only if logprob-based checks are enabled
        self.logprob_confidence_threshold = -2.0      # adjustable
        self.logprob_stable_window = 5               # last K tokens

        # Keywords that signal coherent reasoning
        self.good_keywords = [
            "therefore",
            "thus",
            "in conclusion",
        ]

        # Keywords that signal confusion or error
        self.bad_keywords = [
            "this is wrong",
            "incorrect",
            "mistake",
            "wait",
            "contradiction",
        ]

    # ----------------------------------------------------------------------
    # (A) Keyword heuristic
    # ----------------------------------------------------------------------
    def _keyword_signal(self, partial_cot):
        text = partial_cot.lower()

        for bad in self.bad_keywords:
            if bad in text:
                return "incorrect"

        for good in self.good_keywords:
            # Add simple boundary checks to avoid substring matches
            # e.g. avoid matching "so" in "solution"
            # We look for " word " or "word " at start or " word" at end
            pattern = f" {good} "
            if pattern in text or text.startswith(f"{good} ") or text.endswith(f" {good}"):
                return "correct"

        return "continue"

    # ----------------------------------------------------------------------
    # (B) Logprob stability heuristic
    # ----------------------------------------------------------------------
    def _logprob_signal(self, logprobs):
        if not logprobs or len(logprobs) < self.logprob_stable_window:
            return "continue"

        window = logprobs[-self.logprob_stable_window:]
        mean_lp = np.mean(window)

        if mean_lp > self.logprob_confidence_threshold:
            return "correct"
        return "continue"

    # ----------------------------------------------------------------------
    # (C) Structure heuristic (optional)
    # ----------------------------------------------------------------------
    def _structure_signal(self, partial_cot):
        """
        Simple heuristic: if the reasoning forms a structured pattern
        (e.g., bullet-style or numbered steps), treat it as coherent.
        """
        if any(x in partial_cot for x in ["1.", "2.", "3.", "- "]):
            return "correct"

        return "continue"

    # ----------------------------------------------------------------------
    # Unified evaluation function (used by controller)
    # ----------------------------------------------------------------------
    def evaluate(self, partial_cot, logprobs=None):
        """
        Inputs:
          - partial_cot: reasoning string so far
          - logprobs: list of token logprobs from the last checkpoint

        Output:
          - "correct", "incorrect", or "continue"
        """

        # 1. Keyword heuristic
        verdict = self._keyword_signal(partial_cot)
        if verdict != "continue":
            return verdict

        # 2. Logprob heuristic
        if logprobs is not None:
            verdict = self._logprob_signal(logprobs)
            if verdict != "continue":
                return verdict

        # 3. Structural heuristic
        verdict = self._structure_signal(partial_cot)
        if verdict != "continue":
            return verdict

        return "continue"
