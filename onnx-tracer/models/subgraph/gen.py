import torch
import torch.nn as nn
from torch._higher_order_ops import scan

class BiasAndSum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, xs):
        D = xs.size(-1)

        init_state = torch.zeros(1, D)   # ← rank‑2, not rank‑1
        bias      = nn.Parameter(torch.ones(1, D))   # same rank

        def body(carry, x_t):            # carry, x_t both (1, D)
            y_t   = x_t + bias           # (1, D)
            carry = carry + y_t          # (1, D)
            return carry, y_t.squeeze(0) # carry (1,D) OK, scan‑out (D) fine

        carry, ys = scan(body, init_state, xs)
        return ys, carry


torch.onnx.export(
    BiasAndSum().eval(),
    (torch.randn(4, 1),),
    "network.onnx",
    external_data=False,   # single file
    dynamo=True, 
)
