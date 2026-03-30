import yaml
import numpy as np


class RuleBasedVerifier:
    """
    A rule-based verifier that distinguishes between:
      - "continue"  : reasoning is in progress, no conclusion yet
      - "correct"   : reasoning has reached a clear conclusion
      - "incorrect" : reasoning contains error signals

    Design principles:
      - EOS detection (in cot_generator) is the PRIMARY stop signal.
      - This verifier is a BACKUP for cases where EOS is not emitted.
      - We only return "correct" when explicit conclusion language
        appears (e.g. "the answer is 4 hours").
      - Logprob confidence is NOT used for "correct" — high logprobs
        mean the model is fluent, not that it's finished.
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Strong conclusion signals — these phrases almost always
        # appear at the END of reasoning, not in the middle.
        self.conclusion_keywords = [
            "the answer is",
            "the final answer is",
            "the time taken is",
            "the result is",
            "in conclusion",
        ]

        # Signals that the model is confused or going wrong
        self.bad_keywords = [
            "this is wrong",
            "that's incorrect",
            "i made a mistake",
            "wait, let me",
            "contradiction",
            "this doesn't make sense",
        ]

    # ----------------------------------------------------------------------
    # (A) Keyword heuristic
    # ----------------------------------------------------------------------
    def _keyword_signal(self, partial_cot):
        text = partial_cot.lower()

        for bad in self.bad_keywords:
            if bad in text:
                return "incorrect"

        for good in self.conclusion_keywords:
            if good in text:
                return "correct"

        return "continue"

    # ----------------------------------------------------------------------
    # Unified evaluation
    # ----------------------------------------------------------------------
    def evaluate(self, partial_cot, logprobs=None):
        """
        Returns: "correct", "incorrect", or "continue"

        NOTE: logprobs are accepted for API compatibility but are NOT
        used to determine "correct". High logprobs only mean the model
        is generating fluently, not that reasoning is complete.
        Premature "correct" from logprobs caused the pipeline to exit
        before the model finished calculating.
        """

        verdict = self._keyword_signal(partial_cot)
        return verdict