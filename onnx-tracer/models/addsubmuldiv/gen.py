import torch
import torch.nn as nn

class SimpleModelConst(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(1, 10))  # constant tensor

    def forward(self, x):
        return (x + self.w) * (x - self.w) / (self.w)

model = SimpleModelConst()
model.eval()

x = torch.randn(1, 10)

torch.onnx.export(
    model,
    (x,),
    "network.onnx",
    input_names=["x"],
    output_names=["y"],
    opset_version=11,
    do_constant_folding=True,
)