"""GPU inference, imported only after the Colab dependency cell has run."""
from pathlib import Path
from time import perf_counter
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .core import LABELS, SYSTEM_PROMPT, normalize_scores

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
# Recorded in the successful original training log; do not silently change weights.
BASE_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"


class Router:
    def __init__(self, model, tokenizer, max_tokens=2048):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.max_tokens = max_tokens
        if not tokenizer.chat_template:
            raise ValueError("The saved adapter tokenizer must contain its chat template.")
        if tokenizer.eos_token_id is None:
            raise ValueError("Saved tokenizer has no EOS token.")
        self.candidates = {
            label: tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]
            for label in LABELS
        }

    @classmethod
    def restore(cls, adapter_dir):
        adapter_dir = Path(adapter_dir)
        if not torch.cuda.is_available():
            raise RuntimeError("Choose a T4 GPU runtime before loading the model.")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION,
            torch_dtype=torch.float16, device_map={"": 0},
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        # Training replaced EOS with im_end. Keep restored generation consistent.
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return cls(model, tokenizer)

    def _prompt_ids(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Ticket must contain text.")
        if "<|im_start|>" in text or "<|im_end|>" in text:
            raise ValueError("Ticket contains chat control tokens; send for manual review.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Support Ticket: {text}"},
        ]
        # Do not pass enable_thinking=False: that saved Qwen Jinja template adds
        # an empty think block, whereas qwen3_nothink training used a plain prefix.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if not prompt.endswith("<|im_start|>assistant\n"):
            raise ValueError("Unexpected generation prefix; check the saved tokenizer template.")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + max(map(len, self.candidates.values())) > self.max_tokens:
            raise ValueError("Ticket exceeds the training token limit; manual review required.")
        return ids

    @torch.inference_mode()
    def score(self, text, temperature=1.0):
        prompt = self._prompt_ids(text)
        torch.cuda.synchronize()
        start = perf_counter()
        raw = {}
        for label, completion in self.candidates.items():
            ids = torch.tensor([prompt + completion], device=self.model.device)
            mask = torch.ones_like(ids)
            # Retain only positions that predict the complete label and EOS.
            # Sequential candidates bound peak GPU memory for the large vocabulary.
            logits = self.model(
                input_ids=ids, attention_mask=mask, use_cache=False,
                logits_to_keep=len(completion) + 1,
            ).logits[0, :-1, :].float()
            targets = torch.tensor(completion, device=logits.device)
            token_scores = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            raw[label] = token_scores.sum().item()
            del logits, token_scores
        torch.cuda.synchronize()
        scores = normalize_scores(raw, temperature=temperature)
        return {
            "prediction": max(scores, key=scores.get), "scores": scores,
            "log_scores": raw, "latency_ms": (perf_counter() - start) * 1000,
            "score_method": "Sum of complete-label and EOS log likelihoods",
            "calibrated": False, "temperature": temperature,
        }

    @torch.inference_mode()
    def generate_label(self, text):
        prompt = self._prompt_ids(text)
        ids = torch.tensor([prompt], device=self.model.device)
        output = self.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=24, do_sample=False, use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = self.tokenizer.decode(output[0, len(prompt):], skip_special_tokens=True).strip()
        # Never silently convert a malformed generation into Support general.
        return {"raw_output": generated, "valid": generated in LABELS,
                "prediction": generated if generated in LABELS else None}
