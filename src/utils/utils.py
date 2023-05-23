import torch
import torch.nn as nn


class BatchNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
        transpose=False,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.transpose = transpose

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.dim())
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            dim = input.ndim
            input = input.transpose(dim-1, dim-2)
        output = super().forward(input)
        if self.transpose:
            return output.transpose(dim-1, dim-2)
        else:
            return output


def build_bn_from_ln(ln, transpose=False):
    assert isinstance(ln.normalized_shape, int) or len(ln.normalized_shape) == 1
    num_features = (
        ln.normalized_shape
        if isinstance(ln.normalized_shape, int)
        else ln.normalized_shape[0]
    )
    return BatchNorm(
        num_features=num_features,
        affine=ln.elementwise_affine,
        device=ln.weight.device,
        transpose=transpose,
    )


def replace_module_by_module(module, m_to_replace, build_fn):
    module_output = module
    if isinstance(module, m_to_replace):
        module_output = build_fn(module)
    for name, child in module.named_children():
        module_output.add_module(
            name, replace_module_by_module(child, m_to_replace, build_fn)
        )
    del module
    return module_output


def replace_ln_by_bn2d(module):
    return replace_module_by_module(
        module,
        nn.LayerNorm,
        lambda x: build_bn_from_ln(ln=x, tranpose=False),
    )


def replace_ln_by_bn1d(module):
    return replace_module_by_module(
        module,
        nn.LayerNorm,
        lambda x: build_bn_from_ln(ln=x, transpose=True),
    )

