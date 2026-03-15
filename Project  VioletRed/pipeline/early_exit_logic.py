import yaml


class EarlyExitLogic:
    """
    Centralised early-exit policy.

    Decides:
      - Should we STOP reasoning and move to final answer?
      - Should we CONTINUE generating chain-of-thought?
      - Should we ABORT (incorrect direction)?

    Inputs:
      - verifier verdict: "correct", "incorrect", "continue"
      - current step count
      - total verifier calls so far

    Outputs:
      - "exit"
      - "continue"
      - "abort"
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        pipeline_cfg = self.config["pipeline"]

        self.enable_early_exit = pipeline_cfg["enable_early_exit"]
        self.exit_on_confident_verdict = pipeline_cfg["exit_on_confident_verdict"]
        self.max_verifier_calls = pipeline_cfg["max_verifier_calls"]

    # ----------------------------------------------------------------------
    # Decision function used by controller
    # ----------------------------------------------------------------------
    def decide(self, verifier_verdict, verifier_call_count):
        """
        Inputs:
            verifier_verdict: "correct", "incorrect", or "continue"
            verifier_call_count: number of times verifier has been called

        Output:
            - "exit"
            - "continue"
            - "abort"
        """

        # -----------------------------------------------------------
        # Safety cap: Too many verifier calls → stop & answer now.
        # -----------------------------------------------------------
        if verifier_call_count >= self.max_verifier_calls:
            return "exit"

        # -----------------------------------------------------------
        # If early exit is disabled → always continue
        # -----------------------------------------------------------
        if not self.enable_early_exit:
            return "continue"

        # -----------------------------------------------------------
        # Verifier verdict cases:
        # -----------------------------------------------------------
        if verifier_verdict == "incorrect":
            # Could also choose to continue (your design choice)
            return "abort"

        if verifier_verdict == "correct":
            if self.exit_on_confident_verdict:
                return "exit"
            else:
                return "continue"

        # Default: verifier says "continue"
        return "continue"
