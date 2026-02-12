# src/split_data.py
import json
import random
from pathlib import Path
from typing import List, Dict


from ..src.config_loader import  load_training_config

# def normalize_role(role):
#     if role in ("customer", "user"):
#         return "user"
#     if role in ("agent", "assistant"):
#         return "assistant"
#     # if role == "system":
#     #     return "system"
#     raise ValueError(f"Unknown role: {role}")

def load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into memory."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # record = json.loads(line)
                # for msg in record["messages"]:
                    # normalize_role(msg['role'])
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], path: str):
    """Save records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # -------------------------------
    # Load config
    # -------------------------------
    cfg = load_training_config()

    input_file = cfg["all_data_file"]
    train_file = cfg["train_file"]
    eval_file  = cfg["eval_file"]

    # default 90/10 split if not in config
    train_ratio = cfg.get("train_ratio", 0.9)
    seed = cfg.get("seed_split", 42)

    print("\n⚙️ Using config:")
    print(f"  all_data_file: {input_file}")
    print(f"  train_file:    {train_file}")
    print(f"  eval_file:     {eval_file}")
    print(f"  train_ratio:   {train_ratio}")
    print(f"  seed_split:    {seed}\n")

    # -------------------------------
    # Load dataset
    # -------------------------------
    random.seed(seed)

    input_path = Path(input_file)
    print(f"  input_path: {input_path}")

    assert input_path.exists(), f"❌ Input dataset not found: {input_path}"

    print(f"📥 Loading data from {input_path} ...")
    records = load_jsonl(input_path)
    assert len(records) > 0, "❌ Dataset is empty!"

    print(f"📊 Total examples: {len(records)}")

    # -------------------------------
    # Split
    # -------------------------------
    random.shuffle(records)
    split_idx = int(len(records) * train_ratio)

    train_records = records[:split_idx]
    eval_records  = records[split_idx:]

    print(f"✂️ Split complete:")
    print(f"   Train → {len(train_records)}")
    print(f"   Eval  → {len(eval_records)}\n")

    # -------------------------------
    # Save files
    # -------------------------------

    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_records, train_file)
    save_jsonl(eval_records, eval_file)

    print("✅ JSONL datasets saved successfully!")
    print(f"📁 Train: {train_file}")
    print(f"📁 Eval : {eval_file}\n")


if __name__ == "__main__":
    main()
