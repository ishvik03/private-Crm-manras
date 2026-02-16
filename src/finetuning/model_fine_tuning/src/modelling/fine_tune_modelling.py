# src/modeling.py
"""
Model loading + LoRA config utilities.

This file is responsible for:
1) Loading the base HF model (Mistral 7B Instruct)
2) Optionally quantizing it to 4-bit (to save VRAM)
3) Preparing it for LoRA training
4) Returning a LoRA config that targets the right modules


Common pitfalls:
- Not using quantization -> OOM on most GPUs
- Wrong target_modules -> LoRA doesn't learn
- Forgetting prepare_model_for_kbit_training -> training becomes unstable
"""

from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
torch.cuda.empty_cache()

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
        HF model id, e.g. "mistralai/Mistral-7B-Instruct-v0.2"
    use_4bit: bool
        If True, loads the model in 4-bit quantized mode (recommended for 7B on single GPU)
    f16: bool
        If True and supported, uses bfloat16 compute for stability/perf
    device_map: str
        "auto" places weights across available device(s)
    trust_remote_code: bool
        Leave False unless model repo requires it

    Returns
    -------
    model: AutoModelForCausalLM
        Model ready for LoRA fine-tuning
    """

    # Choose compute dtype for 4-bit quantization math
    # bf16 is best on modern NVIDIA GPUs (A100/H100/4090 etc.). If unsupported, use fp16.
    if f16 :
        compute_dtype = torch.float16
    else :
        compute_dtype = torch.float32


    quant_config = None

    if use_4bit:
        # BitsAndBytes config for 4-bit QLoRA style loading
        # nf4 is typically best; double quant improves compression
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config= quant_config,  # None means full precision
        low_cpu_mem_usage=True,
        torch_dtype= compute_dtype if not use_4bit else None,  # dtype handled by bnb ,if 4bit let BitsAndBytes choose dtype → and do just None for now
        device_map=device_map,
        # trust_remote_code=trust_remote_code,
    )
    # model.to("cpu")


    # IMPORTANT:
    # If training with k-bit weights (4-bit/8-bit), you must "prepare" the model:
    # - enables gradient checkpointing-friendly hooks
    # - casts some layers for stability
    # - makes LoRA training behave correctly on quantized weights
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # This is often recommended for training stability and to avoid cache issues
    model.config.use_cache = False

    return model


def get_lora_config(
    r: int = 16, #last was 8 (this is rank)
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    LoRA config for Microsoft Phi-2 (GPT-NeoX style).

    Returns
    -------
    peft.LoraConfig
    """

    # These module names match Mistral / Llama architecture in HF




    # target_modules = [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "dense",
    # ]


    
    # Check if expected modules exist
    targets_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    


    
    return LoraConfig(
        r=r,                       # rank: higher = more capacity but more VRAM
        lora_alpha=lora_alpha,     # scaling factor; commonly 2*r
        lora_dropout=lora_dropout, # dropout on LoRA weights
        bias=bias,                 # usually "none" for QLoRA
        task_type=task_type,       # causal LM fine-tuning
        target_modules=targets_modules,
    )
