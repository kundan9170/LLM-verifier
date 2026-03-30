import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)


class CoTGenerator:
    """
    Generates chain-of-thought token-by-token with:
    - KV-cache reuse across tokens
    - Multi-EOS detection (Qwen2.5 compatible)
    - Sentence-aware checkpointing: generates until a natural
      sentence boundary, bounded by min/max token limits
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        self.model = model
        self.tokenizer = tokenizer

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]
        self.return_logprobs = self.config["llm"]["return_logprobs"]

        # Sentence-aware checkpoint bounds
        ckpt_cfg = self.config["llm"].get("checkpoint", {})
        self.min_chunk_tokens = ckpt_cfg.get("min_tokens", 10)
        self.max_chunk_tokens = ckpt_cfg.get("max_tokens", 60)

        self.temperature = self.config["generation"]["temperature"]
        self.top_p = self.config["generation"]["top_p"]

        # Pre-build logit processors once
        self.processors = LogitsProcessorList([
            TemperatureLogitsWarper(self.temperature),
            TopPLogitsWarper(self.top_p),
        ])

        # Collect ALL eos token IDs
        self.eos_token_ids = self._collect_eos_ids()

    def _collect_eos_ids(self):
        ids = set()
        tok_eos = self.tokenizer.eos_token_id
        if tok_eos is not None:
            if isinstance(tok_eos, int):
                ids.add(tok_eos)
            else:
                ids.update(tok_eos)
        if hasattr(self.model, "generation_config"):
            gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
            if gen_eos is not None:
                if isinstance(gen_eos, int):
                    ids.add(gen_eos)
                else:
                    ids.update(gen_eos)
        print(f"[CoTGenerator] EOS token IDs: {ids}")
        return ids

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def decode(self, token_id):
        return self.tokenizer.decode(token_id, skip_special_tokens=True)

    # ----------------------------------------------------------------
    # Sentence boundary detection
    # ----------------------------------------------------------------
    def _is_sentence_boundary(self, generated_text, latest_token_text):
        """
        Check if we've reached a natural stopping point.

        We look at the accumulated generated_text (not just the token)
        because a token might be "." but the sentence isn't done yet
        (e.g. "3.14"), or a token might be "\n" which forms a step break.

        Returns True if the text ends at a sentence boundary.
        """
        stripped = generated_text.rstrip()
        if len(stripped) < 5:
            return False

        # Ends with sentence-terminal punctuation
        if stripped.endswith(('.', '?', '!')):
            # Avoid false positives on decimal numbers like "3.14"
            # Check that the char before the dot is NOT a digit
            if stripped.endswith('.') and len(stripped) >= 2:
                char_before = stripped[-2]
                if char_before.isdigit():
                    return False
            return True

        # Ends with a newline after meaningful content (step breaks)
        # e.g. "Step 1: Speed = 13 + 4 = 17 km/hr\n"
        if generated_text.endswith('\n') and len(stripped) > 15:
            return True

        return False

    # ----------------------------------------------------------------
    # Generate next token (with KV-cache)
    # ----------------------------------------------------------------
    def generate_next_token(self, input_ids, past_key_values=None):
        if past_key_values is not None:
            model_input = input_ids[:, -1:]
        else:
            model_input = input_ids

        with torch.no_grad():
            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        new_past = outputs.past_key_values

        if self.return_logprobs:
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            logprobs = None
        next_logprob = None

        logits = self.processors(input_ids, logits)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = torch.clamp(logits, -50, 50)
        logits = logits - logits.max(dim=-1, keepdim=True).values

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1e-8, neginf=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = next_token.item()
        next_token_text = self.decode([next_token_id])

        if logprobs is not None:
            next_logprob = logprobs[0, next_token_id].item()

        new_input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]], device=self.model.device)],
            dim=1,
        )

        hit_eos = (next_token_id in self.eos_token_ids)

        return next_token_id, next_token_text, next_logprob, new_input_ids, new_past, hit_eos

    # ----------------------------------------------------------------
    # Generate until sentence boundary (sentence-aware checkpoint)
    # ----------------------------------------------------------------
    def generate_until_checkpoint(self, prompt_or_ids, past_key_values=None):
        """
        Generates tokens until a natural sentence boundary is reached,
        bounded by min_chunk_tokens and max_chunk_tokens.

        Logic:
          - Always generate at least min_chunk_tokens (don't check boundaries)
          - Between min and max, stop at the first sentence boundary
          - At max_chunk_tokens, stop regardless (hard cap)
          - If EOS is hit at any point, stop immediately

        Returns:
            generated_text, logprobs, input_ids, past_key_values, hit_eos
        """
        if isinstance(prompt_or_ids, str):
            input_ids = self.encode(prompt_or_ids)["input_ids"]
        elif torch.is_tensor(prompt_or_ids):
            input_ids = prompt_or_ids
        elif prompt_or_ids is None:
            raise ValueError("generate_until_checkpoint received None.")
        else:
            raise TypeError(f"Unsupported input type {type(prompt_or_ids)}")

        generated_text = ""
        logprobs = []
        hit_eos = False
        tokens_generated = 0

        for _ in range(self.max_chunk_tokens):
            tok_id, tok_text, tok_logprob, input_ids, past_key_values, eos = (
                self.generate_next_token(input_ids, past_key_values)
            )
            generated_text += tok_text
            tokens_generated += 1

            if tok_logprob is not None:
                logprobs.append(tok_logprob)

            # EOS always stops immediately
            if eos:
                hit_eos = True
                break

            # After min tokens, check for sentence boundary
            if tokens_generated >= self.min_chunk_tokens:
                if self._is_sentence_boundary(generated_text, tok_text):
                    break

        return generated_text, logprobs, input_ids, past_key_values, hit_eos

    # ----------------------------------------------------------------
    # Generate full chain-of-thought (max tokens, no checkpoints)
    # ----------------------------------------------------------------
    def generate_full_cot(self, prompt):
        input_ids = self.encode(prompt)["input_ids"]
        generated_text = ""
        logprobs = []
        past_key_values = None

        for _ in range(self.max_cot_tokens):
            tok_id, tok_text, tok_logprob, input_ids, past_key_values, eos = (
                self.generate_next_token(input_ids, past_key_values)
            )
            generated_text += tok_text
            if tok_logprob is not None:
                logprobs.append(tok_logprob)
            if eos:
                break

        return generated_text, logprobs