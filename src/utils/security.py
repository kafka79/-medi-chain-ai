import os
import base64
import logging
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("medi-chain-security")

def _get_encryption_key() -> bytes:
    key_str = os.getenv("DLQ_ENCRYPTION_KEY")
    if not key_str:
        # Fallback key for testing/dev (never use in real production!)
        return b"dev_key_for_medi_chain_dlq_32B_!"  # Exactly 32 bytes
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
