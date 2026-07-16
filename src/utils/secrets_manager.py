import os
import json
import logging
from typing import Optional

logger = logging.getLogger("secrets-manager")

class SecretsManager:
    """
    Abstracts fetching of sensitive credentials (API keys, encryption keys)
    from an enterprise vault (e.g., AWS Secrets Manager, Azure Key Vault).
    Uses a simple in-memory cache to prevent hammering the vault.
    Falls back to environment variables in TESTING mode.
    """
    _cache = {}

    @classmethod
    def get_secret(cls, secret_name: str) -> Optional[str]:
        if secret_name in cls._cache:
            return cls._cache[secret_name]

        # 1. First, check if we are in testing mode and can use env fallback
        is_testing = os.getenv("TESTING", "false").lower() == "true"
        is_local = os.getenv("STORAGE_MODE", "s3").lower() == "local"

        if is_testing or is_local:
            val = os.getenv(secret_name)
            if val:
                cls._cache[secret_name] = val
                return val

        # 2. Attempt to fetch from AWS Secrets Manager (Boto3)
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            region_name = os.getenv("AWS_REGION", "us-east-1")
            client = boto3.client(service_name='secretsmanager', region_name=region_name)
            
            try:
                # Assuming the vault uses a JSON payload for multiple keys, or a direct string
                # We query a generic "medi_chain/production/secrets" vault
                vault_name = os.getenv("AWS_SECRET_VAULT_NAME", "medi_chain/production/secrets")
                response = client.get_secret_value(SecretId=vault_name)
                
                if 'SecretString' in response:
                    secret_string = response['SecretString']
                    try:
                        secrets_dict = json.loads(secret_string)
                        if secret_name in secrets_dict:
                            cls._cache[secret_name] = secrets_dict[secret_name]
                            return secrets_dict[secret_name]
                    except json.JSONDecodeError:
                        pass
                        
            except ClientError as e:
                logger.error(f"Failed to fetch from AWS Secrets Manager: {e}")
                pass
        except ImportError:
            # Boto3 not installed, continue
            pass

        # 3. Final fallback (which should only trigger in misconfigured prod)
        val = os.getenv(secret_name)
        if val:
            logger.warning(
                f"CRITICAL: Falling back to plaintext environment variable for {secret_name} "
                "in a production-like environment because Vault retrieval failed."
            )
            cls._cache[secret_name] = val
            return val
            
        return None
