import os
import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
from langchain.chat_models.base import BaseChatModel


load_dotenv()

def get_available_models() -> List[str]:
    """
    Fetch available models from OpenRouter API.
    Returns a list of model IDs.
    """
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            # Sort models to put free ones on top or just alphabetically
            # For now, let's just return all IDs sorted alphabetically
            model_ids = [model["id"] for model in models]
            return sorted(model_ids)
        else:
            print(f"Failed to fetch models: {response.status_code} - {response.text}")
            return ["mistralai/mistral-7b-instruct:free", "google/gemini-2.0-flash-lite-preview-02-05:free", "qwen/qwen3-30b-a3b:free"]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["mistralai/mistral-7b-instruct:free", "google/gemini-2.0-flash-lite-preview-02-05:free", "qwen/qwen3-30b-a3b:free"]

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 model_name: str = "mistralai/mistral-7b-instruct:free",
                 **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        # Allow model_name to be overridden by kwargs if present
        model = kwargs.get("model", model_name)
        if "model" in kwargs:
            del kwargs["model"]
            
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            model=model,
            **kwargs
        )
