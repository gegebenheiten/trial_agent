import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from trial_agent.formatting import compact_trial_for_prompt
from trial_agent.agent.prompts import SUGGESTION_TEMPLATE, SYSTEM_PROMPT

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def _load_env() -> None:
    if not load_dotenv:
        return
    current = Path(__file__).resolve()
    for parent in current.parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            break


_load_env()

DEFAULT_BASE_URL = os.getenv("DIFY_BASE_URL", "https://ai-playground.trialos.com.cn/v1")


class DifyClient:
    """
    Minimal wrapper around the Dify chat-messages API.
    Expects an API key in env var DIFY_API_KEY (or passed explicitly).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("DIFY_API_KEY")
        if not self.api_key:
            raise ValueError("DIFY_API_KEY is not set; export it or pass api_key.")

    def chat(
        self,
        prompt: str,
        inputs: Optional[Dict] = None,
        conversation_id: str = "",
        user: str = "trial-agent",
        retries: int = 3,
        timeout: int = 120,
    ) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": inputs or {},
            "query": prompt,
            "response_mode": "blocking",
            "conversation_id": conversation_id,
            "user": user,
        }

        last_error: Optional[str] = None
        for _ in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat-messages",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    return resp.json().get("answer", "")
                last_error = f"{resp.status_code} - {resp.text}"
            except Exception as exc:  # pragma: no cover - network path
                last_error = str(exc)
            time.sleep(2)

        if last_error:
            print(f"[Dify] API error after retries: {last_error}")
        return None


def build_dify_prompt(trial: Dict, retrieved: List[Dict]) -> str:
    """
    Compose a grounded prompt combining the current trial and top-K snippets.
    The downstream Dify app should be configured with the protocol-optimizer instructions.
    """
    import json

    lines = [
        SYSTEM_PROMPT.strip(),
        "\nInstruction:\n" + SUGGESTION_TEMPLATE.strip(),
        "Current trial schema:",
        json.dumps(compact_trial_for_prompt(trial), ensure_ascii=False, indent=2),
        "\nTop similar trials (with snippets):",
    ]
    for item in retrieved:
        payload = {
            "trial_id": item.get("trial_id"),
            "phase": item.get("phase"),
            "primary_endpoint": item.get("primary_endpoint"),
            "snippet": item.get("snippet"),
            "score": item.get("score"),
        }
        trial_compact = item.get("trial_compact")
        if trial_compact:
            payload["trial_compact"] = trial_compact
        lines.append(json.dumps(payload, ensure_ascii=False))
    lines.append(
        "\nGenerate structured recommendations with evidence (design/criteria/endpoints) and medical review questions."
    )
    return "\n".join(lines)
