# src/config_loader.py
import json
from pathlib import Path
from typing import Dict, Any



def load_training_config() -> Dict[str, Any]:
    """
    Load training configuration from a JSON file.
    """

    # Path to current file (config_loader.py)
    current_file = Path(__file__).resolve()

    # Go up TWO levels: modelling → src → model_fine_tuning
    project_root = current_file.parents[1]

    # Now build the config path cleanly
    config_path = project_root / "config" / "training_config.json"

    config_path = Path(config_path)
  
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg
