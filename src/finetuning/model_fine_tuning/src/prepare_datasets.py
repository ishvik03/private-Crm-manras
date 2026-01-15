# src/dataset.py
from typing import Dict, Any, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from .prompts import load_system_prompts, get_prompt_for_issue

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_conversation_text(messages):
    lines = []
    for m in messages:
        content = m["content"].strip()
        lines.append(f"{m["role"]}: {content}")
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
) -> Tuple[Dataset, Dataset, AutoTokenizer]:
    """
    For each example:
      - build system prompt using core_reason/sentiment/priority
      - convert messages[] + system prompt → chat template prompt text
      - append assistant_reasoning as target
      - tokenize into input_ids/attention_mask
    """
    tokenizer = load_tokenizer(model_name)
    prompts = load_system_prompts()  # billing/shipping/technical files

    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_eval  = load_dataset("json", data_files=eval_file,  split="train")



    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        core_reason = example.get("core_reason") or example.get("core_issue")
        sentiment   = example.get("sentiment", "unknown")
        priority    = example.get("priority", "unknown")

        base_prompt = get_prompt_for_issue(core_reason, prompts)

        system_content = (
            "You are an internal reasoning engine."
            # + base_prompt
            # + f"\n\nCustomer sentiment: {sentiment}."
            # + f"\nTicket priority: {priority}."
        )

        system_prompt = {"role": "system", "content": system_content}
        # 1) Visible chat

        # messages = [{"role": "system", "content": system_prompt}] + example["messages"]

        input_text = build_full_prompt(system_prompt, example['messages'])

        target_reasoning = example["agent_reasoning"].strip()

        full_text = input_text + target_reasoning


        # chat_prompt = tokenizer.apply_chat_template(
        #     full_text,
        #     tokenize=False,
        #     add_generation_prompt=True  # tell model: now you respond
        # )

        # 2) Hidden reasoning target

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    train_ds = raw_train.map(preprocess, remove_columns=raw_train.column_names)
    eval_ds  = raw_eval.map(preprocess,  remove_columns=raw_eval.column_names)

    return train_ds, eval_ds, tokenizer
