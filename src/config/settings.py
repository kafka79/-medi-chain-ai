from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class ClinicalThresholds(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLINICAL_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    uncertainty_calibration_factor: float = Field(default=1.0, gt=0.0)
    ood_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    ood_use_static_threshold: bool = Field(default=True)
    ood_cosine_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    ood_text_cosine_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    mc_dropout_passes: int = Field(default=50, ge=10, le=200)

    thresholds_validated: bool = Field(default=False)
    validation_dataset: str = Field(default="")
    validation_date: str = Field(default="")
    validation_metrics: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("thresholds_validated", mode="before")
    @classmethod
    def _parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    def get_audit_dict(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "uncertainty_calibration_factor": self.uncertainty_calibration_factor,
            "ood_confidence_threshold": self.ood_confidence_threshold,
            "ood_use_static_threshold": self.ood_use_static_threshold,
            "ood_cosine_threshold": self.ood_cosine_threshold,
            "ood_text_cosine_threshold": self.ood_text_cosine_threshold,
            "mc_dropout_passes": self.mc_dropout_passes,
            "thresholds_validated": self.thresholds_validated,
            "validation_dataset": self.validation_dataset,
            "validation_date": self.validation_date,
            "validation_metrics": self.validation_metrics,
        }


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    max_workers: int = Field(default=1, ge=1, le=8)
    max_concurrent_requests: int = Field(default=2, ge=1, le=16)
    max_concurrent_requests_fallback: int = Field(default=2, ge=1, le=16)
    internal_api_key: str = Field(default="")
    ssl_verify: bool = Field(default=True)
    ssl_cert_file: Optional[str] = Field(default=None)
    ssl_key_file: Optional[str] = Field(default=None)
    allowed_image_roots: str = Field(default="temp/storage,shared_scans,/app/temp/storage,/app/shared_scans")
    inference_api_url: str = Field(default="http://inference-api:8001")
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    hedging_delay_factor: float = Field(default=1.5, ge=1.0, le=3.0)
    circuit_breaker_failure_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    circuit_breaker_window_seconds: int = Field(default=10, ge=5, le=60)
    circuit_breaker_open_seconds: int = Field(default=30, ge=10, le=300)


class SemaphoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEMAPHORE_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    lease_ttl_seconds: int = Field(default=30, ge=10, le=300)
    reconnect_cooldown_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    max_lease_refresh_retries: int = Field(default=3, ge=1, le=10)


class DriftConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DRIFT_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    key_ttl_seconds: int = Field(default=86400, ge=3600, le=604800)
    min_cases: int = Field(default=50, ge=10, le=1000)
    agreement_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    alert_webhook_url: str = Field(default="")


class PrivacyConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRIVACY_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    ner_model_path: str = Field(default="dslim/bert-base-NER")
    ner_lazy_load: bool = Field(default=False)
    fallback_redaction_aggressive: bool = Field(default=True)


class StorageConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STORAGE_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    mode: str = Field(default="s3")
    minio_endpoint: str = Field(default="minio:9000")
    minio_access_key: str = Field(default="")
    minio_secret_key: str = Field(default="")
    cleanup_max_age_seconds: int = Field(default=3600, ge=300, le=86400)


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    version: str = Field(default="1.3.0")
    max_concurrent_requests: int = Field(default=2, ge=1, le=16)
    api_key: str = Field(default="")
    rate_limit_per_minute: int = Field(default=10, ge=1, le=100)
    enable_background_workers: bool = Field(default=False)
    audit_log_mode: str = Field(default="stdout")
    audit_log_path: str = Field(default="outputs/audit/api_audit.log")


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REDIS_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    url: str = Field(default="redis://redis:6379/0")
    sentinel_hosts: str = Field(default="")
    cluster_nodes: str = Field(default="")
    sentinel_service_name: str = Field(default="mymaster")


class SecurityConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_", 
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    dlq_encryption_key: str = Field(default="")
    dicom_encryption_key: str = Field(default="")
    api_keys_config: str = Field(default="")
    internal_api_key: str = Field(default="")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    testing: bool = Field(default=False)
    storage_mode: str = Field(default="s3")
    log_level: str = Field(default="INFO")
    model_checkpoint: str = Field(default="models/fusion_model.pt")
    ehr_gateway_url: str = Field(default="https://mock-ehr-gateway.internal/fhir")
    milvus_host: str = Field(default="localhost")
    milvus_port: str = Field(default="19530")
    drift_alert_webhook_url: str = Field(default="")
    dlq_dir: str = Field(default="temp/dlq")
    drift_cache_dir: str = Field(default="temp/drift")
    internal_api_key: str = Field(default="")
    api_key: str = Field(default="")


_clinical_thresholds: Optional[ClinicalThresholds] = None
_inference_config: Optional[InferenceConfig] = None
_semaphore_config: Optional[SemaphoreConfig] = None
_drift_config: Optional[DriftConfig] = None
_privacy_config: Optional[PrivacyConfig] = None
_storage_config: Optional[StorageConfig] = None
_api_config: Optional[APIConfig] = None
_redis_config: Optional[RedisConfig] = None
_security_config: Optional[SecurityConfig] = None
_app_settings: Optional[AppSettings] = None


def get_clinical_thresholds() -> ClinicalThresholds:
    global _clinical_thresholds
    if _clinical_thresholds is None:
        _clinical_thresholds = ClinicalThresholds()
        
        calibration_path = Path("config/calibration_report.json")
        if calibration_path.exists():
            try:
                with open(calibration_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                    
                # Use flat values or optimal_thresholds dict
                opt = report.get("optimal_thresholds", report)
                
                if "confidence_threshold" in opt:
                    _clinical_thresholds.confidence_threshold = opt["confidence_threshold"]
                if "uncertainty_threshold" in opt:
                    _clinical_thresholds.uncertainty_threshold = opt["uncertainty_threshold"]
                if "ood_confidence_threshold" in opt:
                    _clinical_thresholds.ood_confidence_threshold = opt["ood_confidence_threshold"]
                if "ood_cosine_threshold" in opt:
                    _clinical_thresholds.ood_cosine_threshold = opt["ood_cosine_threshold"]
                if "ood_text_cosine_threshold" in opt:
                    _clinical_thresholds.ood_text_cosine_threshold = opt["ood_text_cosine_threshold"]
                if "uncertainty_calibration_factor" in opt:
                    _clinical_thresholds.uncertainty_calibration_factor = opt["uncertainty_calibration_factor"]
                if "mc_dropout_passes" in report:
                    _clinical_thresholds.mc_dropout_passes = report["mc_dropout_passes"]
                
                if "validation_metrics" in report:
                    metrics = report["validation_metrics"]
                    _clinical_thresholds.validation_dataset = metrics.get("dataset", "")
                    _clinical_thresholds.validation_date = metrics.get("timestamp", "")
                    _clinical_thresholds.validation_metrics = metrics
                    
                _clinical_thresholds.thresholds_validated = True
            except Exception as e:
                pass

    return _clinical_thresholds


def get_inference_config() -> InferenceConfig:
    global _inference_config
    if _inference_config is None:
        _inference_config = InferenceConfig()
    return _inference_config


def get_semaphore_config() -> SemaphoreConfig:
    global _semaphore_config
    if _semaphore_config is None:
        _semaphore_config = SemaphoreConfig()
    return _semaphore_config


def get_drift_config() -> DriftConfig:
    global _drift_config
    if _drift_config is None:
        _drift_config = DriftConfig()
    return _drift_config


def get_privacy_config() -> PrivacyConfig:
    global _privacy_config
    if _privacy_config is None:
        _privacy_config = PrivacyConfig()
    return _privacy_config


def get_storage_config() -> StorageConfig:
    global _storage_config
    if _storage_config is None:
        _storage_config = StorageConfig()
    return _storage_config


def get_api_config() -> APIConfig:
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config


def get_redis_config() -> RedisConfig:
    global _redis_config
    if _redis_config is None:
        _redis_config = RedisConfig()
    return _redis_config


def get_security_config() -> SecurityConfig:
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def get_app_settings() -> AppSettings:
    global _app_settings
    if _app_settings is None:
        _app_settings = AppSettings()
    return _app_settings


def dump_all_configs() -> Dict[str, Any]:
    return {
        "clinical_thresholds": get_clinical_thresholds().get_audit_dict(),
        "inference": get_inference_config().model_dump(),
        "semaphore": get_semaphore_config().model_dump(),
        "drift": get_drift_config().model_dump(),
        "privacy": get_privacy_config().model_dump(),
        "storage": get_storage_config().model_dump(),
        "api": get_api_config().model_dump(),
        "redis": get_redis_config().model_dump(),
        "security": get_security_config().model_dump(),
        "app": get_app_settings().model_dump(),
    }


def validate_production_config() -> List[str]:
    errors = []
    app = get_app_settings()
    api = get_api_config()
    clinical = get_clinical_thresholds()
    security = get_security_config()
    inference = get_inference_config()

    if not app.testing and app.storage_mode != "local":
        if not api.api_key:
            errors.append("API_KEY must be set in production")
        if not security.dlq_encryption_key:
            errors.append("SECURITY_DLQ_ENCRYPTION_KEY must be set in production")
        if not security.dicom_encryption_key:
            errors.append("SECURITY_DICOM_ENCRYPTION_KEY must be set in production")
        if not inference.internal_api_key:
            errors.append("INFERENCE_INTERNAL_API_KEY must be set in production")
        if not app.internal_api_key:
            errors.append("INTERNAL_API_KEY must be set in production")

    if not clinical.thresholds_validated and not app.testing:
        errors.append("Clinical thresholds not validated (CLINICAL_THRESHOLDS_VALIDATED=false). Set after calibration.")

    return errors


if __name__ == "__main__":
    import sys
    errors = validate_production_config()
    if errors:
        print("CONFIG VALIDATION ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Config validation passed")
        print(json.dumps(dump_all_configs(), indent=2, default=str))