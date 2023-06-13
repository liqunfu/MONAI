# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pydoc import locate
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data.meta_tensor import MetaTensor
from monai.inferers.merger import AvgMerger, Merger
from monai.inferers.splitter import Splitter
from monai.inferers.utils import compute_importance_map
from monai.utils import BlendMode, PatchKeys, PytorchPadMode, ensure_tuple, optional_import
from monai.visualize import CAM, GradCAM, GradCAMpp

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)

import numpy as np

tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

def slices_to_tensor(slices):
    # convert list of list/tuple of python slices to tensor
    tensor_list = []
    for slice_tuple in slices:
        slice_tensor = torch.tensor([[s.start if s.start else 0, s.stop if s.stop else -1] for s in slice_tuple], dtype=torch.int64)
        tensor_list.append(slice_tensor)
    slices_tensor = torch.stack(tensor_list)
    return slices_tensor

def tensor_to_slices(slices_tensor):
    # Convert tensor of shape (N, 3 or 2, 2) to a list of list of Python slices
    slices = []
    for i in range(slices_tensor.size(0)):
        sub_tensor = slices_tensor[i]
        assert sub_tensor.size(0) == 3 or sub_tensor.size(0) == 2
        sub_slices = []
        for j in range(sub_tensor.size(0)):
            sub_sub_tensor = sub_tensor[j]
            assert sub_sub_tensor.size(0) == 2
            start = int(sub_sub_tensor[0])
            stop = int(sub_sub_tensor[0])
            sub_slices.append(slice(start, stop))
        slices.append(sub_slices)
    return slices

def process_predictor_output(seg_prob_out, slices, unravel_slice, importance_map_,
                             output_image_list, count_map_list, image_size, batch_size, roi_size, sw_batch_size,
                             device="cpu", compute_dtype=torch.float32):
    # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
    dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
    
    w_t = importance_map_

    # if len(w_t.shape) == num_spatial_dims:
    #     w_t = w_t[None, None]
    # w_t = w_t.to(dtype=compute_dtype, device=sw_device)

    sw_device_buffer = list(seg_tuple)

    for ss in range(len(sw_device_buffer)):
        b_shape = sw_device_buffer[ss].shape
        seg_chns, seg_shape = b_shape[1], b_shape[2:]
        z_scale = None
        if seg_shape != tuple(roi_size):
            z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
            w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
        if len(output_image_list) <= ss:
            output_shape = [batch_size, seg_chns]
            output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            new_tensor: Callable = torch.zeros  # type: ignore
            output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
            count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
            w_t_ = w_t.to(device)
            for __s in slices:
                if z_scale is not None:
                    __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_

        sw_device_buffer[ss] *= w_t
        sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
        _compute_coords(sw_batch_size, unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])
    sw_device_buffer = []
    return dict_keys

def sw_prepare_win_data_for_predictor_script(inputs, slice_g, num_win, slices, total_slices, sw_batch_size, sw_device="cpu"):
    slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
    unravel_slice = [
        [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
        for idx in slice_range
    ]
    if sw_batch_size > 1:
        win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
    else:
        win_data = inputs[unravel_slice[0]].to(sw_device)
    return win_data, unravel_slice

# this one need jit.script/onnx-script onnx export
def pad_inputs_if_needed_by_roi_size_script(inputs, roi_size, padding_mode, cval):
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)
    return inputs, pad_size


def sw_prepare_for_loop(inputs, roi_size, roi_weight_map, sw_batch_size, overlap, mode, sigma_scale, device="cpu", sw_device=None):
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    for k in range(len(inputs.shape) - 1, 1, -1):
        assert max(roi_size[k - 2] - inputs.shape[k], 0) == 0

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable

    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
            )
            if len(importance_map_.shape) == num_spatial_dims:
                importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

    return windows_range, importance_map_, num_win, total_slices, slices, batch_size, image_size

def sliding_window_inference2(
    inputs: torch.Tensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        process_fn: process inference output and adjust the importance map per window
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    
    _, _, *original_image_size = inputs.shape
    inputs, pad_size = pad_inputs_if_needed_by_roi_size_script(inputs, roi_size, padding_mode, cval)

    windows_range, importance_map_, num_win, total_slices, slices, batch_size, image_size =\
        sw_prepare_for_loop(inputs, roi_size, roi_weight_map, sw_batch_size, overlap, mode, sigma_scale, device, sw_device)

    # stores output and count map
    output_image_list, count_map_list = [], []  # type: ignore
    compute_dtype = torch.float32

    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        win_data, unravel_slice = sw_prepare_win_data_for_predictor_script(inputs, slice_g, num_win, slices, total_slices, sw_batch_size, sw_device)

        seg_prob_out = predictor(win_data, *args, **kwargs)  # batched patch

        dict_keys = process_predictor_output(seg_prob_out, slices, unravel_slice, importance_map_,
                                             output_image_list, count_map_list, image_size, batch_size, roi_size, sw_batch_size,
                                             device="cpu", compute_dtype=torch.float32)

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)


    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        def unpad_to_original_size_script(output_image_list, num_spatial_dims, pad_size, original_image_size, roi_size):
            for ss, output_i in enumerate(output_image_list):
                zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
                final_slicing: list[slice] = []
                for sp in range(num_spatial_dims):
                    si = num_spatial_dims - sp - 1
                    slice_dim = slice(
                        int(round(pad_size[sp * 2] * zoom_scale[si])),
                        int(round((pad_size[sp * 2] + original_image_size[si]) * zoom_scale[si])),
                    )
                    final_slicing.insert(0, slice_dim)
                output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]
            return output_image_list
        num_spatial_dims = len(inputs.shape) - 2
        output_image_list = unpad_to_original_size_script(output_image_list, num_spatial_dims, pad_size, original_image_size, roi_size)


    final_output = _pack_struct(output_image_list, dict_keys)
    final_output = convert_to_dst_type(final_output, inputs, device=device)[0]  # type: ignore
    return final_output  # type: ignore


def _compute_coords(sw, coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)  # type: ignore
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)


class SlidingWindowInferer2(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        roi_size: Sequence[int] | int,
        sw_batch_size: int = 1,
        overlap: Sequence[float] | float = 0.25,
        mode: BlendMode | str = BlendMode.CONSTANT,
        sigma_scale: Sequence[float] | float = 0.125,
        padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
        cpu_thresh: int | None = None,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.progress = progress
        self.cpu_thresh = cpu_thresh

        self.network = network

        # compute_importance_map takes long time when computing on cpu. We thus
        # compute it once if it's static and then save it for future usage
        self.roi_weight_map = None
        try:
            if cache_roi_weight_map and isinstance(roi_size, Sequence) and min(roi_size) > 0:  # non-dynamic roi size
                if device is None:
                    device = "cpu"
                self.roi_weight_map = compute_importance_map(
                    ensure_tuple(self.roi_size), mode=mode, sigma_scale=sigma_scale, device=device
                )
            if cache_roi_weight_map and self.roi_weight_map is None:
                warnings.warn("cache_roi_weight_map=True, but cache is not created. (dynamic roi_size?)")
        except BaseException as e:
            raise RuntimeError(
                f"roi size {self.roi_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e

    def forward(self, inputs):
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """

        device = 'cpu'
        temp_meta = None
        if isinstance(inputs, MetaTensor):
            temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)

        final_output =  sliding_window_inference2(
            inputs,
            self.roi_size,
            self.sw_batch_size,
            self.network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.sw_device,
            device,
            self.progress,
            self.roi_weight_map,
        )

        if temp_meta is not None:
            final_output = MetaTensor(final_output).copy_meta_from(temp_meta)
        return final_output

def register_custom_op(opset_version: int = 12) -> None:
    from torch.onnx import symbolic_helper

    def custom_lift_fresh(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
        return g.op("com.microsoft::lift_fresh")

    @symbolic_helper.parse_args("v", "v", "v", "b")
    def affine_grid_generator(ctx: torch.onnx.SymbolicContext, g, theta, size, align_corners):
        if True:
            align_corners_i = 0 if align_corners else 1
            return g.op(
                "custom::AffineGrid",
                theta,
                size,
                align_corners_i=align_corners_i
                )
        else:
            # this is only for testing. theta shall not be constant in real world.
            theta=torch.tensor([[[9.8824e-01, 0.0000e+00, 0.0000e+00, 5.8824e-03], [0.0000e+00, 1.0030e+00, 0.0000e+00, 4.3355e-04], [0.0000e+00, 0.0000e+00, 1.0030e+00, 4.3355e-04]]], dtype=torch.float64)
            size = [1, 1, 220, 220, 84]
            align_corners = False
            grid = torch.nn.functional.affine_grid(theta=theta, size=size, align_corners=align_corners)
            return g.op(
                "onnx::Constant",
                value_t=torch.tensor(grid.numpy().flatten().tolist())
                )

    def clamp(ctx: torch.onnx.SymbolicContext, g, input, min=None, max=None):
        return g.op(
            "onnx::Clip",
            input,
            min,
            max,
            )

    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::lift_fresh",
        symbolic_fn=custom_lift_fresh,
        opset_version=opset_version,
    )
    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::affine_grid_generator",
        symbolic_fn=affine_grid_generator,
        opset_version=opset_version,
    )

    if opset_version == 20:
        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::clamp",
            symbolic_fn=clamp,
            opset_version=opset_version,
        )

