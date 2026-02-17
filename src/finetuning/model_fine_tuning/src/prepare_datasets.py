# src/dataset.py

from typing import Dict, Any, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import json


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_conversation_text(messages):
    lines = []
    for m in messages:
        role = m.get("role", "").strip().lower()
        content = (m.get("content", "") or "").strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_full_prompt(system_prompt: str, messages):
    conversation_text = build_conversation_text(messages)

    prompt = f"""### SYSTEM
{system_prompt}

### CONVERSATION
{conversation_text}

### TASK
Extract conversation_signals and choose the tool call.

### RESPONSE (JSON ONLY)
"""

    return prompt


def prepare_datasets(
    model_name: str,
    train_file: str,
    eval_file: str,
    max_seq_len: int
) -> Tuple[Dataset, Dataset]:

    tokenizer = load_tokenizer(model_name)

    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_eval = None
    if eval_file:
        raw_eval = load_dataset("json", data_files=eval_file, split="train")

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:

        messages = example["messages"]
        conversation_signals = example.get("conversation_signals")
        target = example.get("target")

        # Build gold output object
        output_obj = {}

        if conversation_signals is not None:
            output_obj["conversation_signals"] = conversation_signals

        if target is not None:
            output_obj["target"] = target

        # ---- SYSTEM PROMPT ----
        system_content = """You extract structured fields and choose a tool call.

Return ONLY valid JSON.
No markdown. No explanations. No extra keys.

Rules:
- Use "unknown" when the conversation does not provide enough info.
- Do not invent details.


Output schema:

{
  "conversation_signals": {
    "issue": {
      "category": "<category>",
      "subtype": "<machine_readable_issue_label>",
      "feature_context": "<optional_feature_or_flow_or_unknown>"
    },
    "reproduction": {
      "frequency": "<frequency>",
      "trigger_steps": [],
      "scope": "<scope>"
    },
    "user_claims": {
      "settings_or_state_claimed_by_user": {},
      "visible_errors": [],
      "behavior_observed": []
    },
    "environment": {
      "platform": "<platform>",
      "app_version_status": "<app_version_status>"
    },
    "troubleshooting_attempted": [],
    "impact_assessment": {
      "severity_hint": "<severity_hint>",
      "customer_sentiment": "<customer_sentiment>"
    }
  },
  "target": {
    "tool_name": "<string>",
    "arguments": {},
    "intent": "<string>",
    "priority": "<priority>"
  }
}
"""

         # ---- Build Prompt Text ----
        prompt_text = build_full_prompt(system_content, messages).strip()

        # ---- Build Completion Text ----
        completion_text = json.dumps(output_obj, ensure_ascii=False)
        
        # ---- TOKENIZE SEPARATELY ----
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
        completion_tokens = tokenizer(completion_text, add_special_tokens=False)

        prompt_ids = prompt_tokens["input_ids"]
        completion_ids = completion_tokens["input_ids"]

        # ---- TRUNCATION LOGIC ----
        # Ensure total length <= max_seq_len
        total_length = len(prompt_ids) + len(completion_ids)

        if total_length > max_seq_len:
            # truncate prompt first (keep completion intact)
            overflow = total_length - max_seq_len
            prompt_ids = prompt_ids[:-overflow]

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        # ---- LABEL MASKING ----
        labels = [-100] * len(prompt_ids) + completion_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ---- Apply mapping ----
    train_ds = raw_train.map(preprocess, remove_columns=raw_train.column_names)

    eval_ds = None
    if raw_eval is not None:
        eval_ds = raw_eval.map(preprocess, remove_columns=raw_eval.column_names)

    return train_ds, eval_ds
