import os
import base64
import logging
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("medi-chain-security")

_EPHEMERAL_KEY = AESGCM.generate_key(bit_length=256)

def _get_encryption_key() -> bytes:
    key_str = os.getenv("DLQ_ENCRYPTION_KEY")
    if not key_str:
        if os.getenv("TESTING") == "true" or os.getenv("STORAGE_MODE") == "local":
            logger.warning(
                "WARNING: DLQ_ENCRYPTION_KEY is not set. Generating a transient ephemeral key for secure local operations. "
                "Note: Local DLQ payloads will not be decryptable across process restarts."
            )
            return _EPHEMERAL_KEY
        raise RuntimeError(
            "CRITICAL: DLQ_ENCRYPTION_KEY environment variable is not set. "
            "Refusing to start with default key fallback in a production-like environment. "
            "Set DLQ_ENCRYPTION_KEY in your environment/secrets."
        )
    try:
        decoded = base64.b64decode(key_str)
        if len(decoded) in [16, 24, 32]:
            return decoded
    except Exception:
        pass
    key_bytes = key_str.encode("utf-8")
    if len(key_bytes) >= 32:
        return key_bytes[:32]
    return key_bytes.ljust(32, b"\0")

def encrypt_payload(payload_str: str) -> str:
    """Encrypts a string payload using AES-GCM and returns a base64 encoded string containing nonce + ciphertext."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    data = payload_str.encode("utf-8")
    ct = aesgcm.encrypt(nonce, data, None)
    return base64.b64encode(nonce + ct).decode("utf-8")

def decrypt_payload(encrypted_b64: str) -> str:
    """Decrypts a base64 encoded string containing nonce + ciphertext using AES-GCM."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    raw = base64.b64decode(encrypted_b64)
    nonce = raw[:12]
    ct = raw[12:]
    data = aesgcm.decrypt(nonce, ct, None)
    return data.decode("utf-8")
