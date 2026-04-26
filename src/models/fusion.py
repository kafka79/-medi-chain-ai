import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionModel(nn.Module):
    """
    Late fusion model for clinical classification.
    Concatenates visual embeddings from BiomedCLIP (512D) 
    and text embeddings from clinical history (768D).
    """
    def __init__(self, vision_dim=512, text_dim=768, hidden_dim=512, output_dim=256, num_classes=5):
        super(LateFusionModel, self).__init__()
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Increased p for better MC Dropout signal
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, vision_emb, text_emb):
        """
        Args:
            vision_emb: Tensor of shape (batch, 512)
            text_emb: Tensor of shape (batch, 768)
        Returns:
            Joint representation (batch, 256) and Logits (batch, num_classes)
        """
        # Concatenate modalities
        combined = torch.cat((vision_emb, text_emb), dim=1)
        
        # Fusion
        joint_repr = self.fusion_layer(combined)
        
        # Classification
        logits = self.classifier(joint_repr)
        
        return joint_repr, logits

    def get_embeddings(self, vision_emb, text_emb):
        """Return only the joint representation for indexing/retrieval."""
        joint_repr, _ = self.forward(vision_emb, text_emb)
        return joint_repr

if __name__ == "__main__":
    model = LateFusionModel()
    v = torch.randn(8, 512)
    t = torch.randn(8, 768)
    joint, logits = model(v, t)
    print(f"Joint representation shape: {joint.shape}")
    print(f"Logits shape: {logits.shape}")
