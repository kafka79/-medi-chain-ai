from opacus import PrivacyEngine
import torch

class PrivateTrainingManager:
    def __init__(self, model, optimizer, data_loader, target_epsilon=8.0, target_delta=1e-5):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        self.privacy_engine = PrivacyEngine()
        
        # Wrap model, optimizer and data loader for differential privacy
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=10, # Example
            max_grad_norm=1.0,
        )
        print(f"Privacy engine initialized. Epsilon budget: {target_epsilon} Delta: {target_delta}")

    def get_privacy_stats(self):
        """Current privacy budget spent."""
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        return {"epsilon": epsilon, "delta": self.target_delta}

    def log_privacy(self):
        stats = self.get_privacy_stats()
        print(f"(ε = {stats['epsilon']:.2f}, δ = {stats['delta']})")

if __name__ == "__main__":
    # Integration test would go here
    pass
