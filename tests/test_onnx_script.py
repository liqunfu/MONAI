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

import os
import shutil
import tempfile
import unittest
from copy import deepcopy

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

import monai
from monai.bundle import ConfigWorkflow
from monai.data import Dataset
from monai.inferers import SimpleInferer, SlidingWindowInferer
from tests.test_onnx_common import (
    SlidingWindowInferer2,
    register_custom_op,
    pad_inputs_if_needed_by_roi_size_script,
    sw_prepare_for_loop,
    sw_prepare_win_data_for_predictor_script,
    slices_to_tensor,
    tensor_to_slices)
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage
from tests.nonconfig_workflow import NonConfigWorkflow

from monai.utils import (BlendMode, PytorchPadMode, ensure_tuple_size)

TEST_CASE_1 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.json")]


TEST_CASE_2 = ["C:/LiqunWA/MONAI/model-zoo-fork/models/brats_mri_segmentation/configs/inference.json"]
TEST_CASE_3 = ["C:/LiqunWA/MONAI/bundles/lung_nodule_ct_detection/configs/inference.json"]
TEST_CASE_4 = ["C:/LiqunWA/MONAI/model-zoo-fork/models/spleen_ct_segmentation/configs/inference.json"]

# TEST_CASE_3 = [os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json")]

import torch
import onnx
import onnxruntime as ort
import io

from onnx import numpy_helper, TensorProto
from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64

from onnxscript.tests.common import onnx_script_test_case

@script()
def maxsum(A: FLOAT["N"], B: FLOAT["N"]) -> FLOAT["N"]:
    sum1 = op.ReduceSum(A)
    sum2 = op.ReduceSum(B)
    if sum1 < sum2:
        result = op.Identity(B)
    else:
        result = op.Identity(A)
    return result

def get_scan_interval_(
    image_size: Sequence[int], roi_size: Sequence[int], overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    scan_interval = []
    for i, o in enumerate(overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

@script()
def get_scan_interval_script(image_size: INT64[3], roi_size: INT64[3], overlap: FLOAT[3]) -> INT64[3]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    ones = op.CastLike(op.Constant(value_int=1), overlap)
    ones_munus_overlap = op.Sub(ones, overlap)
    roi_size_cast = op.CastLike(roi_size, overlap)
    interval_int = op.CastLike(op.Mul(roi_size_cast, ones_munus_overlap), image_size)
    interval = op.Min(op.Max(interval_int, op.Constant(value_int=1)), image_size)
    return interval


def dense_patch_slices(
    image_size: Sequence[int], patch_size: Sequence[int], scan_interval: Sequence[int], return_slice: bool = True
) -> list[tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval
        return_slice: whether to return a list of slices (or tuples of indices), defaults to True

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    if return_slice:
        return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]
    return [tuple((s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]  # type: ignore

@script()
def dense_patch_slices_script(image_size: INT64[3], patch_size: INT64[3], scan_interval: Sequence[int]) -> INT64["N", 3, 2]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.
    """
    # TODO: edge cases when last patch(s) is out of image boundary
    scan_num = op.CastLike(op.Ceil(op.Div(op.Cast(image_size, FLOAT.dtype), op.Cast(scan_interval, FLOAT.dtype))), image_size)

    theta = op.Cast(
        op.Reshape(
        op.Constant(value_floats=[
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,]),
            op.Constant(value_int=[3, 4])), FLOAT.dtype)
    
    starts = op.AffineGrid(theta, op.Reshape(scan_num, op.Constant(value_int=[1, 1, 3])))
    stops = op.Min(op.Add(starts, scan_interval), image_size)
    slices = op.Concat([starts, stops], axis=2)
    return slices

# @script()
def prepare_for_loop(inputs, roi_size, sw_batch_size=1, overlap=0.25):
    num_spatial_dims = len(inputs.shape) - 2
    batch_size, _, *image_size_ = inputs.shape

    # compute all slices
    scan_interval = get_scan_interval_(image_size, roi_size, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range = range(0, total_slices, sw_batch_size)

    return windows_range, num_win, slices, total_slices

# @script()
def prepare_for_predictor(inputs, slice_g, num_win, slices, total_slices, sw_batch_size):
    slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
    unravel_slice = [
        [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
        for idx in slice_range
    ]
    if sw_batch_size > 1:
        win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice])
    else:
        win_data = inputs[unravel_slice[0]].to(sw_device)
    return win_data, unravel_slice

# @script()
def process_predictor_output(seg_tuple, slices, unravel_slice, importance_map,
                             output_image_list, count_map_list, image_size, batch_size):
    # seg_shape and roi_size are of the same shape so no inpolation of weight map is needed.    
    # seg_tuple is a tuple of tensors, each tensor is of shape [seg_chns, *roi_size]

    for ss in range(len(seg_tuple)):
        b_shape = seg_tuple[ss].shape
        seg_chns, seg_shape = b_shape[1], b_shape[2:]
        if len(output_image_list) <= ss:
            output_shape = [batch_size, seg_chns]
            output_shape += list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            new_tensor: Callable = torch.zeros  # type: ignore
            output_image_list.append(new_tensor(output_shape))
            count_map_list.append(torch.zeros([1, 1] + output_shape[2:]))
            for __s in slices:
                count_map_list[-1][(slice(None), slice(None), *__s)] += importance_map

        seg_tuple[ss] *= importance_map
        for original_idx, p in zip(unravel_slice, seg_tuple[ss]):
            idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
            output_image_list[ss][idx_zm] += p

    return output_image_list, count_map_list

# @script()
def sliding_window(
    input: FLOAT["N", "Ci", "D", "H", "W"],
    predictor: Callable[[FLOAT["B", "Ci", "D", "H", "W"]], FLOAT["B", "Co", "D", "H", "W"]],
    roi_size: INT64["D"],
    sw_batch_size: INT64["B"],
    overlap: Sequence[float] | float = 0.25,
    importance_map: FLOAT["N", "H", "W"] | None = None,
) -> FLOAT["N", "C", "H", "W"]:
    # input has been padded so that it is not less than roi_size in all spatial dimensions
    # spatial shape of predictor's input and output are the same 
    # no buffering of sliding window iterations before writing to device memory
    # importance_map is pre-computed
    windows_range, num_win, slices, total_slices = sw_prepare_for_loop(input, roi_size, sw_batch_size, overlap)

    output_image_list = []
    count_map_list = []
    dict_keys: Sequence[str] | None = None,
    for slice_g in windows_range:
        win_data, unravel_slice = prepare_for_predictor(input, slice_g, num_win, slices, total_slices, sw_batch_size)

        seg_prob_out = predictor(win_data)  # how to support *args, **kwargs?

        seg_probs: tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_probs = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            dict_keys = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_probs = tuple(seg_prob_out[k] for k in dict_keys)
        else:
            seg_probs = ensure_tuple(seg_prob_out)  # type: ignore

        output_image_list, count_map_list = process_predictor_output(seg_probs, slices, unravel_slice, importance_map,
                                                                     output_image_list, count_map_list)


    # in the order of applicability return a key to tensor dict, a single tensor, or a tuple of tensors
    if dict_keys is not None:
        return dict(zip(dict_keys, output_image_list))
    if isinstance(output_image_list, (list, tuple)) and len(output_image_list) == 1:
        return output_image_list[0]
    return tuple(output_image_list)


class TestSlidingWindowWithONNXScript(onnx_script_test_case.OnnxScriptTestCase):
    def test_onnx_script(self):
        n = 8
        np.random.seed(0)
        a = np.random.rand(n).astype("float32").T
        b = np.random.rand(n).astype("float32").T

        # FIXME(liqunfu): expected are from ort evaluation.
        # needs numpy oxs to provide expected instead.
        expected = np.array(
            [
                0.5488135,
                0.71518934,
                0.60276335,
                0.5448832,
                0.4236548,
                0.6458941,
                0.4375872,
                0.891773,
            ],
            dtype=np.float32,
        )

        cases = [
            onnx_script_test_case.FunctionTestParams(maxsum, [a, b], [expected])
        ]
        for case in cases:
            # FAIL : Node () Op (local_function) [TypeInferenceError]
            # GraphProto attribute inferencing is not enabled
            # in this InferenceContextImpl instance.
            self.run_converter_test(case)
            self.run_eager_test(case)

    def test_get_scan_interval(self):
        image_size = np.array([10, 10, 10], dtype=np.int64)
        roi_size = np.array([5, 5, 5], dtype=np.int64)
        overlap = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        # def get_scan_interval_(
        #     image_size: Sequence[int], roi_size: Sequence[int], overlap: Sequence[float]
        # ) -> tuple[int, ...]:

        interval = get_scan_interval_(image_size, roi_size, overlap)
        cases = [
            onnx_script_test_case.FunctionTestParams(get_scan_interval_script, [image_size, roi_size, overlap], [interval])
        ]
        for case in cases:
            self.run_converter_test(case)
            self.run_eager_test(case)

        # def dense_patch_slices(
        #     image_size: Sequence[int], patch_size: Sequence[int], scan_interval: Sequence[int], return_slice: bool = True
        # ) -> list[tuple[slice, ...]]:
        slices = dense_patch_slices(image_size, roi_size, interval)
        cases = [
            onnx_script_test_case.FunctionTestParams(dense_patch_slices_script, [image_size, roi_size, interval], [slices])
        ]
        for case in cases:
            self.run_converter_test(case)
            self.run_eager_test(case)


        
if __name__ == "__main__":
    unittest.main()
