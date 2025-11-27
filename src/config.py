from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Manages application settings loaded from environment variables (.env file).
    """

    MODEL_PATH: str = "modelling/artifacts/model.pkl"
    SIZE_ENCODER_PATH: str = "pre_processing/data/artifacts/package_size_encoder.pkl"
    TYPE_ENCODER_PATH: str = "pre_processing/data/artifacts/product_type_encoder.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Create a settings instance
settings = Settings()