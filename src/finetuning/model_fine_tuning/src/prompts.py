# src/prompts.py
from pathlib import Path
from typing import Dict


PROMPT_DIR = Path("/Users/ishaangupta/PycharmProjects/CRMManras/src/finetuning/model_fine_tuning/prompts")

CORE_ISSUE_MAP = {
    # Billing
    "billing": "billing_issue",
    "billing_issue": "billing_issue",

    # Shipping
    "shipping": "shipping_delay",
    "shipping_delay": "shipping_delay",
    "shipping_issue": "shipping_delay",

    # Technical
    "technical": "technical_issue",
    "tech": "technical_issue",
    "tech_issue": "technical_issue",
    "technical_issue": "technical_issue",
}
def load_system_prompts() -> Dict[str, str]:
    """
    Load long system prompts from text files.
    Keys: 'billing', 'shipping', 'technical'
    """
    prompts = {}
    for key in ["billing_issue", "shipping_delay", "technical_issue", "generic"]:
        p = PROMPT_DIR / f"{key}_system_prompt.txt"
        prompts[key] = p.read_text(encoding="utf-8").strip()
    return prompts

def get_prompt_for_issue(core_issue: str, prompts: Dict[str, str]) -> str:
    """
    Map core_issue/core_reason to one of the known prompt keys
    and return the appropriate system prompt text.
    """
    if core_issue is None:
        # default to billing if missing
        return prompts["generic"]

    norm = core_issue.strip().lower()
    mapped = CORE_ISSUE_MAP.get(norm)
    if mapped is None:
        # unknown → fallback
        return prompts["generic"]

    return prompts[mapped]
