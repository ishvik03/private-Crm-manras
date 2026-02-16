# src/dataset.py
from typing import Dict, Any, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from .prompts import load_system_prompts, get_prompt_for_issue
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

def build_full_prompt(system_prompt, messages):
    conversation_text = build_conversation_text(messages)

    prompt = f"""SYSTEM:
            {system_prompt}

            CONVERSATION:
            {conversation_text}

            TASK:
            Analyze the above conversation and produce structured reasoning.

            OUTPUT:
            """

    return prompt




def prepare_datasets(
    model_name: str,
    train_file: str,
    eval_file: str,
    max_seq_len: int
) -> Tuple[Dataset, Dataset]:
    """
    For each example:
      - build system prompt using core_reason/sentiment/priority
      - convert messages[] + system prompt → chat template prompt text
      - append assistant_reasoning as target
      - tokenize into input_ids/attention_mask
    """
    tokenizer = load_tokenizer(model_name)
    # prompts = load_system_prompts()  # billing/shipping/technical files

    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_eval = None
    if eval_file:
      raw_eval = load_dataset("json", data_files=eval_file, split="train")
    



    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        
        messages = example["messages"]
        conversation_signals = example.get("conversation_signals", None)
        target = example.get("target", None)
         
         
        # Combine output fields into one JSON object the model must generate
        # If your dataset only has "target" and not "conversation_signals", adjust here.
         
        output_obj = {}
         
        if conversation_signals is not None:
          output_obj["conversation_signals"] = conversation_signals
         
        if target is not None:
          output_obj["target"] = target

        

        system_content = (
            "You are an assistant that extracts structured conversation_signals and selects a tool call.\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "Follow the schema exactly.\n"
        )

        # system_prompt = {"role": "system", "content": system_content}
        # 1) Visible chat

        # messages_overall = system_prompt + messages
        

        input_text = build_full_prompt(system_content , messages)

        completion = json.dumps(output_obj, ensure_ascii=False)


        full_text = input_text + completion


        # chat_prompt = tokenizer.apply_chat_template(
        #     full_text,
        #     tokenize=False,
        #     add_generation_prompt=True  # tell model: now you respond
        # )

        # 2) Hidden reasoning target

        # tokenized = tokenizer(
        #     full_text,
        #     truncation=True,
        #     max_length=max_seq_len
        # )

        # return {
        #     "input_ids": tokenized["input_ids"],
        #     "attention_mask": tokenized["attention_mask"]
        # }

        return {"text": full_text}

    train_ds = raw_train.map(preprocess, remove_columns=raw_train.column_names)
    eval_ds = raw_eval.map(preprocess, remove_columns=raw_eval.column_names) if raw_eval else None

    return train_ds, eval_ds
