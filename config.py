"""
Configuration module for AI Enforcement Intelligence System.
Handles environment variables and secrets management securely.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Base configuration class."""
    
    # Application settings
    APP_NAME = "AI Enforcement Intelligence System"
    APP_VERSION = "1.0.0"
    
    # Data settings
    DATA_PATH = os.getenv("DATA_PATH", "preprocessed_enforcement_data.csv")
    
    # API Keys (use environment variables, NEVER hardcode)
    API_KEY = os.getenv("API_KEY", None)
    
    # Streamlit settings
    STREAMLIT_THEME = "dark"
    
    # ML Model settings
    ISOLATION_FOREST_CONTAMINATION = 0.08
    ISOLATION_FOREST_RANDOM_STATE = 42
    FORECAST_MONTHS = 6
    
    @classmethod
    def validate(cls):
        """Validate critical configuration settings."""
        if not Path(cls.DATA_PATH).exists():
            raise FileNotFoundError(f"Data file not found: {cls.DATA_PATH}")


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    LOG_LEVEL = "INFO"


def get_config():
    """Get appropriate configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
