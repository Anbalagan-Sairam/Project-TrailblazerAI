# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppConfig(BaseSettings):
    # AWS / Bedrock
    aws_region: str = Field(..., env="AWS_REGION")
    bedrock_embed_model: str = Field(..., env="BEDROCK_EMBED_MODEL")
    bedrock_llm_model: str = Field(..., env="BEDROCK_LLM_MODEL")

    # Pinecone
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_env: str = Field(..., env="PINECONE_ENV")
    pinecone_index_name: str = Field("trailblazeai", env="PINECONE_INDEX_NAME")

    # RAG
    model_k: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )