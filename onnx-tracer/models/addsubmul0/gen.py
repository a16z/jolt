import torch
import torch.nn as nn

class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.tensor([3.14]))  # constant scalar

    def forward(self, x):
        return (x + self.w) * (x - self.w)

model = ScalarModel()
model.eval()

x = torch.tensor([2.0])  # scalar input (1D tensor)

torch.onnx.export(
    model,
    (x,),
    "network.onnx",
    input_names=["x"],
    output_names=["y"],
    opset_version=11,
    do_constant_folding=True,
)