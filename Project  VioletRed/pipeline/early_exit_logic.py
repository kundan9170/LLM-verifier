import yaml


class EarlyExitLogic:
    """
    Centralised early-exit policy.

    Decides:
      - "exit"     : stop reasoning, produce final answer
      - "continue" : keep generating chain-of-thought
      - "abort"    : reasoning went wrong, give up

    Now includes a minimum-calls guard so we don't exit on the
    very first "correct" signal (the model might just be starting).
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        pipeline_cfg = self.config["pipeline"]

        self.enable_early_exit = pipeline_cfg["enable_early_exit"]
        self.exit_on_confident_verdict = pipeline_cfg["exit_on_confident_verdict"]
        self.max_verifier_calls = pipeline_cfg["max_verifier_calls"]

        # Minimum checkpoints before we allow an early exit.
        # This prevents exiting after just 1 checkpoint when the
        # model has barely started reasoning.
        # Default to 2 if not set in config.
        self.min_calls_before_exit = pipeline_cfg.get("min_verifier_calls_before_exit", 2)

    def decide(self, verifier_verdict, verifier_call_count):
        """
        Inputs:
            verifier_verdict:   "correct", "incorrect", or "continue"
            verifier_call_count: number of times verifier has been called

        Output:
            "exit", "continue", or "abort"
        """

        # Safety cap: too many verifier calls → stop now
        if verifier_call_count >= self.max_verifier_calls:
            return "exit"

        # Early exit disabled → always continue
        if not self.enable_early_exit:
            return "continue"

        # Incorrect → abort
        if verifier_verdict == "incorrect":
            return "abort"

        # Correct → exit only if enabled AND we've generated enough
        if verifier_verdict == "correct":
            if self.exit_on_confident_verdict and verifier_call_count >= self.min_calls_before_exit:
                return "exit"
            else:
                return "continue"

        # Default: continue
        return "continue"
