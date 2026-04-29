"""
Model definitions for DeepPrime PE6 inference.
Contains GeneInteractionModelVanilla and the ensemble wrapper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneInteractionModelVanilla(nn.Module):
    """Vanilla DeepPrime model matching the genet package's original structure."""

    def __init__(
        self,
        c1_in_channels=4,
        c1_out_channels=128,
        c1_kernel_size=(2, 3),
        c1_stride=1,
        c1_padding=(0, 1),
        c1_batchnorm_features=128,
        hidden_size=128,
        num_layers=1,
        num_features=24,
        dropout=0.1,
        c2_out_channels=108,
        c2_kernel_size=3,
        c2_stride=1,
        c2_padding=1,
        r_in_features=128,
        s_out_features=12,
        d_hidden1=96,
        d_hidden2=64,
        d_out_features=128,
        head_in_features=140,
        head_out_features=1,
    ):
        super().__init__()
        s_in_features = 2 * hidden_size
        d_in_features = num_features

        self.c1 = nn.Sequential(
            nn.Conv2d(c1_in_channels, c1_out_channels, c1_kernel_size, c1_stride, c1_padding),
            nn.BatchNorm2d(c1_batchnorm_features),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(c1_out_channels, c2_out_channels, c2_kernel_size, c2_stride, c2_padding),
            nn.BatchNorm1d(c2_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2_out_channels, c2_out_channels, c2_kernel_size, c2_stride, c2_padding),
            nn.BatchNorm1d(c2_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2_out_channels, c1_out_channels, c2_kernel_size, c2_stride, c2_padding),
            nn.BatchNorm1d(c1_out_channels),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.r = nn.GRU(r_in_features, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.s = nn.Linear(s_in_features, s_out_features, bias=False)
        self.d = nn.Sequential(
            nn.Linear(d_in_features, d_hidden1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden1, d_hidden2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden2, d_out_features, bias=False),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_in_features),
            nn.Dropout(dropout),
            nn.Linear(head_in_features, head_out_features, bias=True),
        )

    def forward(self, g, x):
        g = torch.squeeze(self.c1(g), 2)
        g = self.c2(g)
        g, _ = self.r(torch.transpose(g, 1, 2))
        g = self.s(g[:, -1, :])
        x = self.d(x)
        return F.softplus(self.head(torch.cat((g, x), dim=1)))


class ParallelDeepPrimeModels(nn.Module):
    """Ensemble of DeepPrime models applied in parallel."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, g, x):
        return [model(g, x) for model in self.models]


class EnsembleModel(nn.Module):
    """Wraps ParallelDeepPrimeModels to match checkpoint state_dict structure.

    State dict keys: feature_extractor.models.{i}.{layer}
    """

    def __init__(self, num_ensemble=20):
        super().__init__()
        models = [
            GeneInteractionModelVanilla(
                c1_in_channels=4, c1_out_channels=128, c1_kernel_size=[2, 3],
                c1_stride=1, c1_padding=[0, 1], c1_batchnorm_features=128,
                hidden_size=128, num_layers=1, num_features=24, dropout=0.1,
                c2_out_channels=108, c2_kernel_size=3, c2_stride=1, c2_padding=1,
                r_in_features=128, s_out_features=12, d_hidden1=96, d_hidden2=64,
                d_out_features=128, head_in_features=140, head_out_features=1,
            )
            for _ in range(num_ensemble)
        ]
        self.feature_extractor = ParallelDeepPrimeModels(models)

    def forward(self, g, b):
        outputs = self.feature_extractor(g, b)
        return torch.mean(torch.stack(outputs, dim=0), dim=0)

    @classmethod
    def from_checkpoint(cls, ckpt_path, device="cpu"):
        if hasattr(torch.serialization, "add_safe_globals"):
            import functools
            torch.serialization.add_safe_globals([functools.partial])
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        # Infer num_ensemble from checkpoint
        num_ensemble = max(
            int(k.split(".")[2]) for k in state_dict if k.startswith("feature_extractor.models.")
        ) + 1
        model = cls(num_ensemble=num_ensemble)
        # Load only feature_extractor keys
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if k.startswith("feature_extractor.")},
            strict=True,
        )
        model.to(device)
        model.eval()
        return model
