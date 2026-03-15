import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMModelLoader:
    """
    Loads the LLM + tokenizer using HuggingFace Transformers.
    """

    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)

        self.model_name = self.config["llm"]["model_name"]
        self.load_dtype = self.config["llm"]["load_dtype"]
        self.device = self._select_device(self.config["llm"]["device"])
        self.return_logprobs = self.config["llm"].get("return_logprobs", False)

        self.tokenizer = None
        self.model = None

    # ---------------------------------------------------------
    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ---------------------------------------------------------
    def _select_device(self, device_setting):
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting

    # ---------------------------------------------------------
    def _dtype_from_string(self):
        if self.load_dtype == "float16":
            return torch.float16
        elif self.load_dtype == "bfloat16":
            return torch.bfloat16
        elif self.load_dtype == "float32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.load_dtype}")

    # ---------------------------------------------------------
    def load_model(self):
        print(f"[ModelLoader] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        # =============================
        # Important fix for Phi-2
        # =============================
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[ModelLoader] Loading model: {self.model_name}")
        torch_dtype = self._dtype_from_string()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )

        # Also fix pad token on model side
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print("[ModelLoader] Model and tokenizer loaded successfully.")
        return self.model, self.tokenizer

    # ---------------------------------------------------------
    def get(self):
        if self.model is None or self.tokenizer is None:
            return self.load_model()
        return self.model, self.tokenizer