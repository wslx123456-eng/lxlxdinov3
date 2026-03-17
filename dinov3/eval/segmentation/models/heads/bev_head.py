import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVSegHead(nn.Module):
    def __init__(
        self,
        in_channels,
        n_output_channels,
        hidden_dim=256,
        use_batchnorm=True,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        norm = nn.SyncBatchNorm if use_batchnorm else nn.Identity
        self.fuse = nn.Sequential(
            nn.Conv2d(self.channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            norm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            norm(hidden_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden_dim, n_output_channels, kernel_size=1)

    def _transform_inputs(self, inputs):
        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        return torch.cat(inputs, dim=1)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(list(inputs))
        x = self.dropout(x)
        return self.fuse(x)

    def forward(self, inputs):
        return self.classifier(self._forward_feature(inputs))

    def predict(self, x, rescale_to=(512, 512)):
        x = self.classifier(self._forward_feature(x))
        return F.interpolate(input=x, size=rescale_to, mode="bilinear", align_corners=False)
