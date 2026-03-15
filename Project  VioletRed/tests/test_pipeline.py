import unittest
from unittest.mock import MagicMock

# Import your pipeline modules
from llm_engine.cot_generator import CoTGenerator
from llm_engine.final_answer_generator import FinalAnswerGenerator
from verifier.rule_based_verifier import RuleBasedVerifier
from pipeline.early_exit_logic import EarlyExitLogic
from pipeline.controller import PipelineController


# ------------------------------------------------------------
# Mock Model + Tokenizer for Testing
# ------------------------------------------------------------
class DummyModel:
    """
    A minimal dummy HF-like model,
    returns predictable tokens for stable testing.
    """

    def __init__(self):
        self.device = "cpu"

    def generate(self, **kwargs):
        # Always generate a fixed answer token sequence
        return [[0, 1, 2, 3, 4]]

    def __call__(self, input_ids, use_cache=True, output_logits=False):
        # Return fake logits of length vocab=10
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        logits = MagicMock()
        logits.logits = (0.1 * torch.randn(batch, seq_len, 10))  # random logits
        return logits


class DummyTokenizer:
    """
    Minimal tokenizer for tests.
    """

    def __init__(self):
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"

    def encode(self, text):
        return [1, 2, 3]  # fake token IDs

    def decode(self, token_ids, skip_special_tokens=True):
        return "dummy_token"

    def __call__(self, text, return_tensors="pt"):
        import torch
        return {"input_ids": torch.tensor([[1, 2, 3]])}


# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------
class TestLLMPipeline(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer()

    # --------------------------------------------------------
    def test_cot_generator(self):
        gen = CoTGenerator(self.model, self.tokenizer)
        text, logprobs, _ = gen.generate_until_checkpoint("Hello")
        self.assertIsInstance(text, str)

    # --------------------------------------------------------
    def test_final_answer_generator(self):
        ans_gen = FinalAnswerGenerator(self.model, self.tokenizer)
        result = ans_gen.generate_final_answer("Reasoning: therefore 42.")
        self.assertTrue(len(result) > 0)

    # --------------------------------------------------------
    def test_rule_based_verifier(self):
        verifier = RuleBasedVerifier()
        verdict = verifier.evaluate("This is correct because math works.")
        self.assertIn(verdict, ["correct", "incorrect", "continue"])

    # --------------------------------------------------------
    def test_early_exit_logic(self):
        logic = EarlyExitLogic()
        decision = logic.decide("correct", 1)
        self.assertIn(decision, ["exit", "continue", "abort"])

    # --------------------------------------------------------
    def test_pipeline_controller(self):
        controller = PipelineController(self.model, self.tokenizer)
        answer = controller.run("What is 2+2?")
        self.assertIsInstance(answer, str)


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    import torch
    unittest.main()
