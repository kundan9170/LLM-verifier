import yaml
import torch
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)


class CoTGenerator:
    """
    Generates chain-of-thought token-by-token for Qwen3 thinking mode.

    Key features:
    - KV-cache reuse across all tokens
    - Multi-EOS detection (Qwen3 compatible)
    - </think> token detection (marks transition from CoT to answer)
    - Token injection: can force </think> into the sequence
    - Sentence-aware checkpointing
    """

    def __init__(self, model, tokenizer, config_path="config/config.yaml"):
        self.model = model
        self.tokenizer = tokenizer

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.max_cot_tokens = self.config["llm"]["max_cot_tokens"]
        self.return_logprobs = self.config["llm"]["return_logprobs"]

        ckpt_cfg = self.config["llm"].get("checkpoint", {})
        self.min_chunk_tokens = ckpt_cfg.get("min_tokens", 10)
        self.max_chunk_tokens = ckpt_cfg.get("max_tokens", 60)

        self.temperature = self.config["generation"]["temperature"]
        self.top_p = self.config["generation"]["top_p"]
        self.top_k = self.config["generation"].get("top_k", 20)

        # Pre-build logit processors
        processors = [
            TemperatureLogitsWarper(self.temperature),
            TopPLogitsWarper(self.top_p),
        ]
        if self.top_k > 0:
            processors.append(TopKLogitsWarper(self.top_k))
        self.processors = LogitsProcessorList(processors)

        # Collect EOS and </think> token IDs
        self.eos_token_ids = self._collect_eos_ids()
        self.think_end_token_id = self._find_think_end_token()

    # ----------------------------------------------------------------
    # Token ID collection
    # ----------------------------------------------------------------
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

    def _find_think_end_token(self):
        """
        Find the token ID for </think>.
        Qwen3 has this as a special token in the vocabulary.
        """
        # Try encoding </think> — for Qwen3 it should be a single token
        think_end_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)

        if len(think_end_tokens) == 1:
            token_id = think_end_tokens[0]
            print(f"[CoTGenerator] </think> token ID: {token_id}")
            return token_id

        # If it's multiple tokens, store the full sequence
        # and we'll detect by text matching instead
        decoded = self.tokenizer.decode(think_end_tokens)
        print(f"[CoTGenerator] </think> encodes to {len(think_end_tokens)} tokens: {think_end_tokens}")
        print(f"[CoTGenerator] Will use text matching for </think> detection")
        return None

    # ----------------------------------------------------------------
    # Encode / Decode
    # ----------------------------------------------------------------
    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def decode(self, token_id):
        return self.tokenizer.decode(token_id, skip_special_tokens=False)

    # ----------------------------------------------------------------
    # Sentence boundary detection
    # ----------------------------------------------------------------
    def _is_sentence_boundary(self, generated_text, latest_token_text):
        stripped = generated_text.rstrip()
        if len(stripped) < 5:
            return False

        if stripped.endswith(('.', '?', '!')):
            if stripped.endswith('.') and len(stripped) >= 2:
                if stripped[-2].isdigit():
                    return False
            return True

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
        hit_think_end = (
            self.think_end_token_id is not None
            and next_token_id == self.think_end_token_id
        )

        return (next_token_id, next_token_text, next_logprob,
                new_input_ids, new_past, hit_eos, hit_think_end)

    # ----------------------------------------------------------------
    # Inject tokens into the sequence (for forcing </think>)
    # ----------------------------------------------------------------
    def inject_tokens(self, text, input_ids, past_key_values):
        """
        Inject arbitrary text into the token stream.
        Runs forward passes to update the KV-cache, as if the model
        had generated these tokens itself.

        Used by the controller to inject '</think>\n\n' when the
        verifier decides reasoning is complete.

        Returns: (updated_input_ids, updated_past_key_values)
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if not token_ids:
            return input_ids, past_key_values

        # Process all injected tokens in one forward pass
        inject_tensor = torch.tensor([token_ids], device=self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inject_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )

        new_past = outputs.past_key_values

        # Append injected tokens to full input_ids
        new_input_ids = torch.cat([input_ids, inject_tensor], dim=1)

        return new_input_ids, new_past

    # ----------------------------------------------------------------
    # Generate until sentence boundary (thinking phase)
    # ----------------------------------------------------------------
    def generate_until_checkpoint(self, prompt_or_ids, past_key_values=None):
        """
        Generates tokens until a sentence boundary, EOS, or </think>.

        Returns:
            generated_text, logprobs, input_ids, past_key_values,
            hit_eos, hit_think_end
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
        hit_think_end = False
        tokens_generated = 0

        for _ in range(self.max_chunk_tokens):
            (tok_id, tok_text, tok_logprob, input_ids, past_key_values,
             eos, think_end) = self.generate_next_token(input_ids, past_key_values)

            generated_text += tok_text
            tokens_generated += 1

            if tok_logprob is not None:
                logprobs.append(tok_logprob)

            if eos:
                hit_eos = True
                break

            if think_end:
                hit_think_end = True
                break

            if tokens_generated >= self.min_chunk_tokens:
                if self._is_sentence_boundary(generated_text, tok_text):
                    break

        return generated_text, logprobs, input_ids, past_key_values, hit_eos, hit_think_end

    # ----------------------------------------------------------------
    # Generate until EOS (answering phase — no checkpoints)
    # ----------------------------------------------------------------
    def generate_until_eos(self, input_ids, past_key_values, max_tokens=200):
        """
        After </think>, generate the final answer until EOS.
        No verifier checks — just let the model finish.
        """
        generated_text = ""

        for _ in range(max_tokens):
            (tok_id, tok_text, tok_logprob, input_ids, past_key_values,
             eos, think_end) = self.generate_next_token(input_ids, past_key_values)

            generated_text += tok_text

            if eos:
                break

        return generated_text.strip(), input_ids, past_key_values