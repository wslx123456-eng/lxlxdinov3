import torch
import torch.nn as nn


class InputChannelAdapter(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, mode: str = "identity"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode

        if mode == "identity":
            if in_channels != out_channels:
                raise ValueError("identity input adapter requires matching channel counts")
            self.proj = None
        elif mode == "repeat":
            if in_channels != 1:
                raise ValueError("repeat input adapter only supports single-channel inputs")
            self.proj = None
        elif mode == "conv":
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            self.reset_parameters()
        else:
            raise ValueError(f"Unsupported input adapter mode: {mode}")

    def reset_parameters(self):
        if self.proj is None:
            return
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)
        if self.in_channels == 1 and self.out_channels == 3:
            with torch.no_grad():
                self.proj.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return x
        if self.mode == "repeat":
            return x.repeat(1, self.out_channels, 1, 1)
        return self.proj(x)
