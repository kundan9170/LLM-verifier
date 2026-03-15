import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CoTGenerator:
    """
    Generates chain-of-thought token-by-token.
    Provides utilities to:
    - generate_next_token()
    - generate_until_checkpoint()
    - generate_full_cot()
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        self.model = model
        self.tokenizer = tokenizer

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]
        self.checkpoint_interval = self.config["llm"]["checkpoint_interval"]
        self.return_logprobs = self.config["llm"]["return_logprobs"]

        self.temperature = self.config["generation"]["temperature"]
        self.top_p = self.config["generation"]["top_p"]

    # ----------------------------------------------------------------
    # Encode
    # ----------------------------------------------------------------
    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    # ----------------------------------------------------------------
    # Decode
    # ----------------------------------------------------------------
    def decode(self, token_id):
        return self.tokenizer.decode(token_id, skip_special_tokens=True)

    # ----------------------------------------------------------------
    # Generate next token from model
    # ----------------------------------------------------------------
    def generate_next_token(self, input_ids):
        """
        Takes current input_ids and generates the next token + logprob.
        Returns:
        - next_token_id
        - next_token_text
        - logprob (if enabled)
        - new input_ids
        """

        # ---------------------------------------------------------
        # 1. Forward pass
        # ---------------------------------------------------------
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            output_logits=self.return_logprobs,
        )

        logits = outputs.logits[:, -1, :]  # last-token logits

        # ---------------------------------------------------------
        # 2. Logprobs BEFORE warping (for debugging/verifier)
        # ---------------------------------------------------------
        if self.return_logprobs:
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            logprobs = None
        next_logprob = None

        # ---------------------------------------------------------
        # 3. Apply HF warpers (temperature + top-p nucleus sampling)
        # ---------------------------------------------------------
        from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

        processors = LogitsProcessorList([
            TemperatureLogitsWarper(self.temperature),
            TopPLogitsWarper(self.top_p),
        ])
        logits = processors(input_ids, logits)

        # ---------------------------------------------------------
        # 4. Stabilise logits BEFORE softmax   <-- CRITICAL FOR PHI-2
        # ---------------------------------------------------------
        # Replace NaNs / infs
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Clip logits (prevents explosions)
        logits = torch.clamp(logits, -50, 50)

        # Subtract max for numerical stability
        logits = logits - logits.max(dim=-1, keepdim=True).values

        # ---------------------------------------------------------
        # 5. Convert logits → probabilities
        # ---------------------------------------------------------
        probs = torch.softmax(logits, dim=-1)

        # Clean probabilities again (rare but safe)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1e-8, neginf=1e-8)

        # Re-normalize to sum to 1.0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # ---------------------------------------------------------
        # 6. Sample next token safely
        # ---------------------------------------------------------
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()
        next_token_text = self.decode([next_token_id])

        # Compute logprob AFTER sampling
        if logprobs is not None:
            next_logprob = logprobs[0, next_token_id].item()

        # ---------------------------------------------------------
        # 7. Append sampled token ID to input_ids
        # ---------------------------------------------------------
        new_input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]], device=self.model.device)],
            dim=1
        )

        return next_token_id, next_token_text, next_logprob, new_input_ids

    # ----------------------------------------------------------------
    # Generate until next checkpoint
    # ----------------------------------------------------------------
    def generate_until_checkpoint(self, prompt_or_ids):
        """
        Accepts either:
        - a string prompt (first call)
        - existing input_ids tensor (subsequent calls)
        """

        # ---------------------------------------------------------
        # Identify input source
        # ---------------------------------------------------------
        if isinstance(prompt_or_ids, str):
            # First call
            input_ids = self.encode(prompt_or_ids)["input_ids"]

        elif torch.is_tensor(prompt_or_ids):
            # Subsequent calls
            input_ids = prompt_or_ids

        elif prompt_or_ids is None:
            # If None is given, this is a controller bug — do not crash
            raise ValueError(
                "generate_until_checkpoint received None. "
                "Controller must pass either a string prompt or input_ids tensor."
            )

        else:
            raise TypeError(
                f"Unsupported input type {type(prompt_or_ids)} "
                "for generate_until_checkpoint()."
            )

        # ---------------------------------------------------------
        # Generate tokens for one checkpoint
        # ---------------------------------------------------------
        generated_text = ""
        logprobs = []

        for _ in range(self.checkpoint_interval):
            tok_id, tok_text, tok_logprob, input_ids = self.generate_next_token(input_ids)
            generated_text += tok_text
            if tok_logprob is not None:
                logprobs.append(tok_logprob)

        return generated_text, logprobs, input_ids

    # ----------------------------------------------------------------
    # Generate full chain-of-thought (max tokens)
    # ----------------------------------------------------------------
    def generate_full_cot(self, prompt):
        input_ids = self.encode(prompt)["input_ids"]
        generated_text = ""
        logprobs = []

        for _ in range(self.max_cot_tokens):
            tok_id, tok_text, tok_logprob, input_ids = self.generate_next_token(input_ids)
            generated_text += tok_text
            if tok_logprob is not None:
                logprobs.append(tok_logprob)

        return generated_text, logprobs
