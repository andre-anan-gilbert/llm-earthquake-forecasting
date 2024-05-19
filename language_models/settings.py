"""Global settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Class that implements the app settings."""

    GROQ_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
