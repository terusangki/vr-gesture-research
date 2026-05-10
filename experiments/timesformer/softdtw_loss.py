import torch
import torch.nn as nn

class SoftDTW(nn.Module):
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y shape: (Sequence_Length, Dimensions)
        x = x.float()
        y = y.float()

        T1, D = x.shape
        T2, _ = y.shape

        # 거리 계산
        dist = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0) ** 2
        
        gamma = self.gamma
        inf = float("inf")
        R = x.new_full((T1 + 1, T2 + 1), inf)
        R[0, 0] = 0.0

        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                r0 = -R[i - 1, j] / gamma
                r1 = -R[i, j - 1] / gamma
                r2 = -R[i - 1, j - 1] / gamma
                maxi = torch.max(torch.stack([r0, r1, r2]))
                rsum = torch.exp(r0 - maxi) + torch.exp(r1 - maxi) + torch.exp(r2 - maxi)
                softmin = -gamma * (torch.log(rsum) + maxi)
                R[i, j] = dist[i - 1, j - 1] + softmin

        return R[T1, T2]
