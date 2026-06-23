import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json

def load_diagnostic_classes():
    # Attempt to load from environment variable first
    env_classes = os.getenv("DIAGNOSTIC_CLASSES")
    if env_classes:
        try:
            parsed = json.loads(env_classes)
            if isinstance(parsed, list) and len(parsed) > 0:
                return [str(c) for c in parsed]
        except Exception:
            # If not valid JSON, split by comma
            return [c.strip() for c in env_classes.split(",") if c.strip()]
            
    # Attempt to load from config file config/classes.json
    config_path = os.getenv("DIAGNOSTIC_CLASSES_PATH", "config/classes.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
        except Exception as e:
            print(f"Warning: Failed to load diagnostic classes from {config_path}: {e}")

    # Fallback to defaults
    return ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]

DIAGNOSTIC_CLASSES = load_diagnostic_classes()
NUM_CLASSES = len(DIAGNOSTIC_CLASSES)


def get_model_num_classes() -> int:
    """Return the number of output classes the model is configured for.

    This allows the Web API container to validate that DIAGNOSTIC_CLASSES
    matches the model's expected output dimension WITHOUT instantiating
    the full LateFusionModel (which would needlessly load PyTorch weights
    on a container that should remain GPU-free).
    """
    return NUM_CLASSES

class AttentionFusion(nn.Module):
    """
    Addresses Flaw #5: Overengineered Multimodal Fusion.
    Implements a first-principles true cross-modal Multi-Head Attention layer.
    Retains norm1, norm2, and ffn properties for backward compatibility with testing suites.
    """
    def __init__(self, vision_dim=512, text_dim=768, hidden_dim=512, num_classes=NUM_CLASSES):
        super(AttentionFusion, self).__init__()
        
        # Projection layers to align dimensions
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        
        # True self-attention block over the multimodal sequence [v, t]
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Standard Transformer Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, vision_emb, text_emb):
        # Project both to a common semantic space
        v = self.v_proj(vision_emb)  # Shape: (batch, hidden_dim)
        t = self.t_proj(text_emb)    # Shape: (batch, hidden_dim)
        
        # Stack to form a multimodal sequence of length 2: Shape (batch, 2, hidden_dim)
        fused_seq = torch.stack([v, t], dim=1)
        
        # True self-attention where visual and text features attend to each other
        attn_out, _ = self.self_attn(query=fused_seq, key=fused_seq, value=fused_seq)
        
        # Aggregate the attended tokens back to single representation (average pooling)
        fused = attn_out.mean(dim=1)  # Shape: (batch, hidden_dim)
        
        # Additive residual connection and first layer norm
        fused = self.norm1(fused + (v + t) / 2.0)
        
        # Feed-forward refinement and second layer norm (with residual connection)
        fused = self.norm2(fused + self.ffn(fused))
        
        logits = self.classifier(fused)
        return fused, logits

# Keep LateFusionModel for backward compatibility or rename it
class LateFusionModel(AttentionFusion):
    pass

if __name__ == "__main__":
    model = LateFusionModel()
    v = torch.randn(8, 512)
    t = torch.randn(8, 768)
    joint, logits = model(v, t)
    print(f"Joint representation shape: {joint.shape}")
    print(f"Logits shape: {logits.shape}")
