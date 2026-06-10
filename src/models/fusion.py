import torch
import torch.nn as nn
import torch.nn.functional as F

DIAGNOSTIC_CLASSES = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
NUM_CLASSES = len(DIAGNOSTIC_CLASSES)

class AttentionFusion(nn.Module):
    """
    Addresses Tara's 'Oversimplification' critique.
    Implements a standard Transformer Encoder block with a Feed-Forward Network
    and double LayerNorm to align visual and text embeddings before final classification.
    """
    def __init__(self, vision_dim=512, text_dim=768, hidden_dim=512, num_classes=NUM_CLASSES):
        super(AttentionFusion, self).__init__()
        
        # Projection layers to align dimensions
        self.v_proj = nn.Linear(vision_dim, hidden_dim)
        self.t_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
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
        v = self.v_proj(vision_emb).unsqueeze(1)  # Shape: (batch, 1, hidden_dim)
        t = self.t_proj(text_emb).unsqueeze(1)    # Shape: (batch, 1, hidden_dim)
        
        # Concatenate tokens to form a sequence of length 2 (multimodal attention)
        seq = torch.cat([v, t], dim=1)            # Shape: (batch, 2, hidden_dim)
        
        # Multi-head self-attention
        attn_output, _ = self.attention(seq, seq, seq) # Shape: (batch, 2, hidden_dim)
        
        # First residual connection and LayerNorm
        seq = self.norm1(seq + attn_output)
        
        # Feed-Forward Network block
        ffn_output = self.ffn(seq)
        
        # Second residual connection and LayerNorm
        seq = self.norm2(seq + ffn_output)
        
        # Average pooling along the sequence dimension to obtain a single joint representation
        fused = seq.mean(dim=1)                   # Shape: (batch, hidden_dim)
        
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
