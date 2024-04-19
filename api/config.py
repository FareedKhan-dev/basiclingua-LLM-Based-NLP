from pydantic_settings import BaseSettings
import os

DEV_MODE = True if os.environ.get("DEV_MODE", "0") == "1" else False

class Settings(BaseSettings):
    gemini_api_key: str

settings = Settings(_env_file=".env", _env_file_encoding="utf-8")

