import redis
from code.config import app_config


db_connection = redis.Redis(
    host=app_config.REDIS_URL,
    port=19211,
    password=app_config.REDIS_PASSWORD,
    decode_responses=True,
)
