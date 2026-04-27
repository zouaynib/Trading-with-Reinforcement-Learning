import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Approximates Q(s, a) for all 3 actions at once.

    Input  : state vector of size (state_dim,)
    Output : Q-values of size (3,)  — one per action (hold, buy, sell)
    """

    def __init__(self, state_dim: int, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    import torch
    net = QNetwork(state_dim=22)

    # Single state
    state = torch.randn(22)
    q_values = net(state)
    print("Input shape :", state.shape)
    print("Output shape:", q_values.shape)
    print("Q-values    :", q_values.detach().numpy().round(4))

    # Batch of 64
    batch = torch.randn(64, 22)
    out = net(batch)
    print("Batch output:", out.shape)
    print("network.py OK")
