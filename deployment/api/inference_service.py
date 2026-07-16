import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib

from fastapi import FastAPI, UploadFile, File, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import secrets
import torch
import uvicorn
import shutil
import tempfile
import asyncio
import base64
import cv2
import concurrent.futures

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vlm.visual_encoder import BiomedVisualEncoder
from sentence_transformers import SentenceTransformer
from src.models.fusion import LateFusionModel
from src.vlm.explainability import VisualExplainer
from src.config.settings import get_inference_config

# Thread pool for inference
INFERENCE_MAX_WORKERS = int(os.getenv("INFERENCE_MAX_WORKERS", "1"))
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=INFERENCE_MAX_WORKERS)
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "models/fusion_model.pt")


def _file_sha256(path: str):
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceService:
    def __init__(self):
        self.encoder = BiomedVisualEncoder()
        self.text_encoder = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        
        self.fusion = LateFusionModel()
        checkpoint_path = MODEL_CHECKPOINT
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # Copy v_proj and t_proj weights to gates if gates are missing in checkpoint
            if "v_gate.weight" not in state_dict and "v_proj.weight" in state_dict:
                state_dict["v_gate.weight"] = state_dict["v_proj.weight"].clone()
                state_dict["v_gate.bias"] = state_dict["v_proj.bias"].clone()
            if "t_gate.weight" not in state_dict and "t_proj.weight" in state_dict:
                state_dict["t_gate.weight"] = state_dict["t_proj.weight"].clone()
                state_dict["t_gate.bias"] = state_dict["t_proj.bias"].clone()
            self.fusion.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {checkpoint_path} (backward-compatible mode)")
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Using random weights.")
            
        self.fusion = self.fusion.to(self.encoder.device)
        self.fusion.eval()
        from src.models.uncertainty import UncertaintyEstimator
        self.uncertainty = UncertaintyEstimator(self.fusion)
        self.visual_explainer = VisualExplainer(self.encoder.model, self.encoder.preprocess)

    def run_visual_inference_sync(self, img_path: str):
        """Helper to run visual encoding and Grad-CAM generation synchronously inside a thread pool."""
        from PIL import Image, ImageEnhance
        
        try:
            with Image.open(img_path) as pil_img:
                perturbed_imgs = []
                for i in range(5):
                    if i == 0:
                        perturbed_imgs.append(pil_img.convert("RGB"))
                    elif i == 1:
                        perturbed_imgs.append(pil_img.rotate(3).convert("RGB"))
                    elif i == 2:
                        perturbed_imgs.append(pil_img.rotate(-3).convert("RGB"))
                    elif i == 3:
                        perturbed_imgs.append(pil_img.transform(pil_img.size, Image.Transform.AFFINE, (1, 0, 4, 0, 1, 4)).convert("RGB"))
                    elif i == 4:
                        enhancer = ImageEnhance.Brightness(pil_img)
                        perturbed_imgs.append(enhancer.enhance(0.95).convert("RGB"))
                
                features_batch = self.encoder.encode_image(perturbed_imgs)
                # Keep original features as main prediction features
                features = features_batch[0]
                # Compute visual standard deviation across perturbed inputs
                visual_std = torch.std(features_batch, dim=0)
        except Exception as e:
            print(f"TTA visual encoding failed ({e}). Falling back to standard encoding.")
            features = self.encoder.encode_image(img_path)[0]
            visual_std = torch.zeros_like(features)
        
        # Generate heatmap
        heatmap_base64 = ""
        try:
            visualization = self.visual_explainer.generate_heatmap(img_path)
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', visualization_bgr)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Failed to generate heatmap: {e}")
            
        return features.tolist(), visual_std.tolist(), heatmap_base64


service = None


def _allowed_image_roots() -> List[Path]:
    configured = os.getenv(
        "INFERENCE_ALLOWED_IMAGE_ROOTS",
        "temp/storage,shared_scans,/app/temp/storage,/app/shared_scans",
    )
    roots = []
    for raw_root in configured.split(","):
        raw_root = raw_root.strip()
        if not raw_root:
            continue
        root = Path(raw_root)
        if not root.is_absolute():
            root = Path.cwd() / root
        roots.append(root.resolve(strict=False))
    return roots


def _path_is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_local_image_path(image_path: str) -> str:
    if not image_path:
        raise HTTPException(400, "image_path is required.")

    try:
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(400, f"Image path {image_path} does not exist on local disk.")
    except OSError as exc:
        raise HTTPException(400, f"Invalid image path: {exc}")

    if not resolved.is_file():
        raise HTTPException(400, "image_path must point to a file.")

    if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}:
        raise HTTPException(415, "Unsupported image path type. Use JPEG, PNG, or DICOM.")

    allowed_roots = _allowed_image_roots()
    if allowed_roots and not any(_path_is_under(resolved, root) for root in allowed_roots):
        raise HTTPException(403, "Image path is outside allowed inference roots.")

    return str(resolved)


def get_service(request: Request) -> InferenceService:
    active_service = getattr(request.app.state, "service", None) or service
    if not active_service:
        raise HTTPException(503, "Models not loaded")
    return active_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    # Flaw #2 (partial): Fail at startup if INTERNAL_API_KEY is not set in production
    if os.getenv("TESTING") != "true" and not os.getenv("INTERNAL_API_KEY"):
        raise RuntimeError("INTERNAL_API_KEY environment variable is required. Cannot start with default credentials.")
    service = InferenceService()
    app.state.service = service
    try:
        yield
    finally:
        app.state.service = None
        service = None


app = FastAPI(
    title="MEdi Chain AI - Inference Microservice",
    description="Dedicated GPU-bound microservice for raw AI inference, decoupling heavy PyTorch models from the Web API.",
    lifespan=lifespan,
)

class TextPayload(BaseModel):
    text: str

class ImagePathPayload(BaseModel):
    image_path: str

class EstimatePayload(BaseModel):
    visual_features: Any
    visual_std: Optional[List[float]] = None
    text_features: Any
    num_passes: int = 50

INTERNAL_API_KEY_HEADER = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

async def verify_internal_api_key(api_key: str = Security(INTERNAL_API_KEY_HEADER)):
    expected_key = os.getenv("INTERNAL_API_KEY")
    # Flaw #2 Fix: No hardcoded default — if env var is missing, reject all requests
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration: INTERNAL_API_KEY not set.")
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid Internal API Key")
    return api_key


@app.post("/encode/image")
async def encode_image(
    request: Request,
    image: UploadFile = File(...),
    api_key: str = Depends(verify_internal_api_key)
):
    active_service = get_service(request)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name
        request.state.upload_path = tmp_path
        
    try:
        loop = asyncio.get_running_loop()
        features_list, visual_std_list, heatmap_base64 = await loop.run_in_executor(
            _inference_executor, active_service.run_visual_inference_sync, tmp_path
        )
                
        return {
            "features": features_list,
            "visual_std": visual_std_list,
            "heatmap_base64": heatmap_base64
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/encode/image_path")
async def encode_image_path(
    request: Request,
    payload: ImagePathPayload,
    api_key: str = Depends(verify_internal_api_key)
):
    active_service = get_service(request)
    
    img_path = validate_local_image_path(payload.image_path)
    request.state.image_path = img_path
        
    loop = asyncio.get_running_loop()
    features_list, visual_std_list, heatmap_base64 = await loop.run_in_executor(
        _inference_executor, active_service.run_visual_inference_sync, img_path
    )
        
    return {
        "features": features_list,
        "visual_std": visual_std_list,
        "heatmap_base64": heatmap_base64
    }


@app.post("/encode/text")
async def encode_text(
    request: Request,
    payload: TextPayload,
    api_key: str = Depends(verify_internal_api_key)
):
    active_service = get_service(request)
    
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(
        _inference_executor, active_service.text_encoder.encode, [payload.text]
    )
    return {"embeddings": emb.tolist()}


@app.post("/estimate")
async def estimate_uncertainty(
    request: Request,
    payload: EstimatePayload,
    api_key: str = Depends(verify_internal_api_key)
):
    import numpy as np
    active_service = get_service(request)
        
    v = torch.tensor(payload.visual_features, dtype=torch.float32, device=active_service.encoder.device)
    if v.ndim == 1:
        v = v.unsqueeze(0)
        
    t = torch.tensor(payload.text_features, dtype=torch.float32, device=active_service.encoder.device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    
    visual_std = None
    if payload.visual_std is not None:
        visual_std = torch.tensor(payload.visual_std, dtype=torch.float32, device=active_service.encoder.device)
        
    # Run uncertainty estimation in thread pool
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        _inference_executor, active_service.uncertainty.estimate_uncertainty, v, t, payload.num_passes, visual_std
    )
        
    # Convert numpy/torch arrays to lists for JSON
    clean_results = {}
    for k, val in results.items():
        if isinstance(val, (torch.Tensor, np.ndarray)):
            clean_results[k] = val.tolist()
        else:
            clean_results[k] = val
            
    return clean_results


@app.get("/health")
async def health_check(request: Request):
    return {
        "status": "ok",
        "models_loaded": getattr(request.app.state, "service", None) is not None,
    }


@app.get("/health/gpu")
async def gpu_health():
    """GPU health endpoint for monitoring."""
    try:
        if not torch.cuda.is_available():
            return {"gpu_available": False, "message": "CUDA not available"}
        
        device_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = props.total_memory / (1024 ** 3)
            free = total - reserved
            
            gpu_info.append({
                "device": i,
                "name": props.name,
                "total_memory_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(free, 2),
                "utilization_pct": round((reserved / total) * 100, 1),
            })
        
        return {
            "gpu_available": True,
            "device_count": device_count,
            "gpus": gpu_info,
            "cuda_version": torch.version.cuda,
        }
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}


@app.get("/version")
async def get_version():
    """Expose model version metadata for audit trail."""
    return {
        "service": "inference-api",
        "version": "1.3.0",
        "model_checkpoint": MODEL_CHECKPOINT,
        "checkpoint_sha256": _file_sha256(MODEL_CHECKPOINT),
        "visual_encoder": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "text_encoder": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    }


if __name__ == "__main__":
    ssl_keyfile = os.getenv("INFERENCE_SSL_KEYFILE", None)
    ssl_certfile = os.getenv("INFERENCE_SSL_CERTFILE", None)
    ssl_ca_certs = os.getenv("INFERENCE_SSL_CA_CERTS", None)
    ssl_cert_reqs = int(os.getenv("INFERENCE_SSL_CERT_REQS", "0"))
    
    kwargs = {}
    if ssl_keyfile and ssl_certfile:
        kwargs["ssl_keyfile"] = ssl_keyfile
        kwargs["ssl_certfile"] = ssl_certfile
        if ssl_ca_certs:
            kwargs["ssl_ca_certs"] = ssl_ca_certs
            kwargs["ssl_cert_reqs"] = ssl_cert_reqs
            
    uvicorn.run(app, host="0.0.0.0", port=8001, **kwargs)