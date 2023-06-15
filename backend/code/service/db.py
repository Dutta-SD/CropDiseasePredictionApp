import uuid
from typing import Tuple
from code.db.connection import db_connection
from code.config import app_config


def generate_key_token() -> Tuple[str, str]:
    """Generates new key value pair for validation

    Returns:
        Tuple[str, str]: key, value pair
    """
    # TODO: Add logic for key and value generation
    key = uuid.uuid4()
    value = uuid.uuid4()
    return str(key), str(value)


def store_in_db(key: str, value: str) -> str | None:
    """Stores value in DB with key

    Args:
        key (str): key to store in redis
        value (str): value to store

    Returns:
        str: key which is stored
    """
    has_been_set = db_connection.set(
        name=key,
        value=value,
        ex=app_config.REDIS_TTL,
    )
    if has_been_set:
        return key
    return None


def check_in_db(key: str) -> bool:
    """Check if key in db or not

    Args:
        key (str): key to check in DB

    Returns:
        bool: key present or not
    """
    return db_connection.exists(key) > 0


def delete_from_db(key: str) -> bool:
    """Deletes from DB

    Args:
        key (str): key to delete

    Returns:
        bool: deleted then true, not found then false
    """
    num_deleted = db_connection.delete(key)
    return True if num_deleted > 0 else False
