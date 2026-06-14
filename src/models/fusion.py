import torch
import torch.nn as nn
import torch.nn.functional as F

DIAGNOSTIC_CLASSES = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
NUM_CLASSES = len(DIAGNOSTIC_CLASSES)

class AttentionFusion(nn.Module):
    """
    Addresses Flaw #5: Overengineered Multimodal Fusion.
    Replaces the mathematically excessive sequence-length-2 Self-Attention block
    with a first-principles projection and cross-modal gating mechanism.
    Retains norm1, norm2, and ffn properties for backward compatibility with testing suites.
    """
    def __init__(self, vision_dim=512, text_dim=768, hidden_dim=512, num_classes=NUM_CLASSES):
        super(AttentionFusion, self).__init__()
        
        # Projection layers to align dimensions
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gating networks
        self.v_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.t_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
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
        
        # Cross-modal gating:
        # The visual representations are gated by the context of text, and vice versa
        gated_v = v * self.t_gate(t)
        gated_t = t * self.v_gate(v)
        
        # Additive fusion and first layer norm
        fused = self.norm1(gated_v + gated_t)
        
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
