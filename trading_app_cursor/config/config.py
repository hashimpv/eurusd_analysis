import os

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "trading_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
}

API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

BASE_URL = "https://www.alphavantage.co/query"
