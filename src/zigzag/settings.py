import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    tavily_api_key: str
    openai_base_url: str
    openai_api_key: str
    openai_model_name: str = "deepseek-chat"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding="utf-8"
    )


settings = Settings()
