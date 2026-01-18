# src/prompts.py
from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent.parent
#this moves you OUT of the inner `src/`

DEFAULT_PROMPT_DIR = BASE_DIR / "prompts"

PROMPT_DIR = Path(os.getenv("PROMPT_DIR", DEFAULT_PROMPT_DIR))

if not PROMPT_DIR.exists():
    raise FileNotFoundError(f"Prompt directory not found: {PROMPT_DIR}")

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
