#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp


def get_filter_torch(*args, **kwargs):
    import warnings
    import torch
    import torch.nn as nn

    def _select_torch_device():
        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            device_cc = torch.cuda.get_device_capability(0)
            supported_arches = set(torch.cuda.get_arch_list())
            required_arch = f"sm_{device_cc[0]}{device_cc[1]}"
            if supported_arches and required_arch not in supported_arches:
                warnings.warn(
                    f"PyTorch build does not support CUDA arch {required_arch}; falling back to CPU for traversability filter."
                )
                return torch.device("cpu")
        except Exception as exc:
            warnings.warn(
                f"Failed to validate CUDA capability ({exc}); falling back to CPU for traversability filter."
            )
            return torch.device("cpu")

        return torch.device("cuda")

    class TraversabilityFilter(nn.Module):
        def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False):
            super(TraversabilityFilter, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
            self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
            self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
            self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

            # Set weights.
            self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
            self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
            self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
            self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

        def __call__(self, elevation_cupy):
            # Convert cupy tensor to pytorch.
            elevation_cupy = elevation_cupy.astype(cp.float32)
            device = self.conv1.weight.device
            if device.type == "cuda":
                elevation = torch.as_tensor(elevation_cupy, device=device)
            else:
                elevation = torch.from_numpy(cp.asnumpy(elevation_cupy)).to(device)

            with torch.no_grad():
                out1 = self.conv1(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
                out2 = self.conv2(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
                out3 = self.conv3(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))

                out1 = out1[:, :, 2:-2, 2:-2]
                out2 = out2[:, :, 1:-1, 1:-1]
                out = torch.cat((out1, out2, out3), dim=1)
                # out = F.concat((out1, out2, out3), axis=1)
                out = self.conv_out(out.abs())
                out = torch.exp(-out)
                if out.is_cuda:
                    out_cupy = cp.from_dlpack(out)
                else:
                    out_cupy = cp.asarray(out.numpy())

            return out_cupy

    device = _select_torch_device()
    traversability_filter = TraversabilityFilter(*args, **kwargs).to(device).eval()
    return traversability_filter


def get_filter_chainer(*args, **kwargs):
    import os

    os.environ["CHAINER_WARN_VERSION_MISMATCH"] = "0"
    import chainer
    import chainer.links as L
    import chainer.functions as F

    class TraversabilityFilter(chainer.Chain):
        def __init__(self, w1, w2, w3, w_out, use_cupy=True):
            super(TraversabilityFilter, self).__init__()
            self.conv1 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=1, nobias=True, initialW=w1)
            self.conv2 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=2, nobias=True, initialW=w2)
            self.conv3 = L.Convolution2D(1, 4, ksize=3, pad=0, dilate=3, nobias=True, initialW=w3)
            self.conv_out = L.Convolution2D(12, 1, ksize=1, nobias=True, initialW=w_out)

            if use_cupy:
                self.conv1.to_gpu()
                self.conv2.to_gpu()
                self.conv3.to_gpu()
                self.conv_out.to_gpu()
            chainer.config.train = False
            chainer.config.enable_backprop = False

        def __call__(self, elevation):
            out1 = self.conv1(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
            out2 = self.conv2(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))
            out3 = self.conv3(elevation.reshape(-1, 1, elevation.shape[0], elevation.shape[1]))

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = F.concat((out1, out2, out3), axis=1)
            out = self.conv_out(F.absolute(out))
            return F.exp(-out).array

    traversability_filter = TraversabilityFilter(*args, **kwargs)
    return traversability_filter


if __name__ == "__main__":
    import cupy as cp
    from parameter import Parameter

    elevation = cp.random.randn(202, 202, dtype=cp.float32)
    print("elevation ", elevation.shape)
    param = Parameter()
    fc = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
    print("chainer ", fc(elevation))

    ft = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
    print("torch ", ft(elevation))
