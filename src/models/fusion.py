import torch
import torch.nn as nn
import torch.nn.functional as F

DIAGNOSTIC_CLASSES = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
NUM_CLASSES = len(DIAGNOSTIC_CLASSES)

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
        
        # Cross-modal multi-head attention blocks
        self.cross_attn_v2t = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.cross_attn_t2v = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
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
        
        # Prepare for MultiheadAttention by adding a sequence dimension of length 1
        # Shape: (batch, 1, hidden_dim)
        v_seq = v.unsqueeze(1)
        t_seq = t.unsqueeze(1)
        
        # Cross-modal attention:
        # Vision query attends to Text key/value, and vice-versa
        attn_v, _ = self.cross_attn_v2t(query=v_seq, key=t_seq, value=t_seq)
        attn_t, _ = self.cross_attn_t2v(query=t_seq, key=v_seq, value=v_seq)
        
        # Squeeze sequence dimension back
        gated_v = attn_v.squeeze(1)
        gated_t = attn_t.squeeze(1)
        
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
