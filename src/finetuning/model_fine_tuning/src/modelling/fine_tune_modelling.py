# src/modeling.py
"""
Model loading + LoRA config utilities.

This file is responsible for:
1) Loading the base HF model
2) Optionally quantizing it to 4-bit (to save VRAM)
3) Preparing it for LoRA training
4) Returning a LoRA config that targets the right modules

Common pitfalls:
- Not using quantization -> OOM on most GPUs
- Wrong target_modules -> LoRA doesn't learn
- Forgetting prepare_model_for_kbit_training -> training becomes unstable
"""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training


def load_base_model(
    model_name: str,
    use_4bit: bool = True,
    f16: bool = True,
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> AutoModelForCausalLM:
    """
    Load the base model for SFT + LoRA.

    Parameters
    ----------
    model_name: str
        HF model id, e.g. "microsoft/phi-2"
    use_4bit: bool
        If True, loads the model in 4-bit quantized mode
    f16: bool
        If True, uses fp16 compute (bf16 is preferred on supported GPUs)
    device_map: str
        "auto" places weights across available device(s)
    trust_remote_code: bool
        Leave False unless model repo requires it

    Returns
    -------
    model: AutoModelForCausalLM
        Model ready for LoRA fine-tuning
    """

    # Choose compute dtype for 4-bit quantization math.
    # On many Colab GPUs (T4), fp16 is the safe default.
    compute_dtype = torch.float16

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,  # None means full precision
        low_cpu_mem_usage=True,
        torch_dtype=compute_dtype if not use_4bit else None,  # dtype handled by bnb in 4-bit mode
        device_map=device_map,
        # trust_remote_code=trust_remote_code,
    )

    # IMPORTANT:
    # If training with k-bit weights (4-bit/8-bit), you must "prepare" the model:
    # - adds hooks that play well with grad checkpointing
    # - casts certain layers for stability
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Training stability / memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    LoRA config (target modules must match your model architecture).

    Returns
    -------
    peft.LoraConfig
    """

    # These module names are common for Mistral/LLaMA-style attention blocks.
    # NOTE: Phi-2 may not use these exact names; verify via:
    #   [n for n, _ in model.named_modules() if any(k in n for k in ["q_proj","k_proj","v_proj","dense"])]
    target_modules = [
    "query_key_value",
    "dense",
      ]

    # ✅ CRITICAL BUG FIX:
    # You previously defined `target_modules` but returned `targets_modules` (typo / mismatch).
    # That would crash or silently misconfigure LoRA depending on your code path.
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
    )
