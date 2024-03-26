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
import math

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
from monai.utils import CompInitMode, ensure_tuple, first, instantiate, optional_import, run_debug, run_eval
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
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
from onnxscript.onnx_opset import opset18 as op
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

def make_test_config_workflow_inferer(config_file):
    override = {
        "network": "$@network_def.to(@device)",
        "dataset#_target_": "Dataset",
        "postprocessing#transforms#2#output_postfix": "seg",
        "output_dir": "c:/temp/monai",
    }
    # test standard MONAI model-zoo config workflow
    inferer = ConfigWorkflow(
        workflow="infer",
        config_file=config_file,
        logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
        **override,
    )
    return inferer

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

def slices_as_ndarray(slices):
    """
    Convert a list of slices to a tensor.

    Args:
        slices: an array of slices objects. the array is of shape (num_patches, num_dims)

    Returns:
        a numpy array of shape (num_patches, num_dims, 2)

    """
    return np.asarray([[x.start, x.stop] for x in np.asarray(slices).reshape(-1)]).reshape(len(slices), -1, 2)

@script()
def dense_patch_slices_script(image_size: INT64[3], patch_size: INT64[3], scan_interval: INT64[3]) -> INT64["N", 3, 2]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.
    """
    # TODO: edge cases when last patch(s) is out of image boundary
    scan_num = op.CastLike(op.Ceil(op.Div(op.Cast(image_size - patch_size, FLOAT.dtype), op.Cast(scan_interval, FLOAT.dtype))), image_size) + 1
    D, H, W = op.Split(scan_num, num_outputs=3)
    step_D, step_H, step_W = op.Split(scan_interval, num_outputs=3)
    zeros_D_H_W = op.CastLike(op.ConstantOfShape(scan_num), image_size)
    
    grid_w_0 = op.Range(0, W * step_W, step_W)
    grid_w = grid_w_0 + zeros_D_H_W

    grid_h_0 = op.Range(0, H * step_H, step_H)
    zeros_D_W_H = op.Transpose(zeros_D_H_W, perm=[0, 2, 1])
    grid_h_1 = zeros_D_W_H + grid_h_0
    grid_h = op.Transpose(grid_h_1, perm=[0, 2, 1])

    grid_d_0 = op.Range(0, D * step_D, step_D)
    zeros_H_W_D = op.Transpose(zeros_D_H_W, perm=[1, 2, 0])
    grid_d_1 = zeros_H_W_D + grid_d_0
    grid_d = op.Transpose(grid_d_1, perm=[2, 0, 1])
                        
    original_grid_start_seq = op.SequenceConstruct(grid_d, grid_h, grid_w)
    original_grid_start_stack = op.ConcatFromSequence(original_grid_start_seq, axis=-1, new_axis=1)  # [D, H, W, 3]
    original_grid_start = op.Reshape(original_grid_start_stack, op.Constant(value_ints=[-1, 3]))  # [D*H*W, 3]
    original_grid_start = op.Min(original_grid_start, image_size - patch_size)
    original_grid_stop = original_grid_start + patch_size
    original_grid_seq = op.SequenceConstruct(original_grid_start, original_grid_stop)
    slices = op.ConcatFromSequence(original_grid_seq, axis=-1, new_axis=1)  # [D*H*W, 3, 2]
    return slices

def prepare_for_loop(inputs, roi_size, sw_batch_size=1, overlap=0.25):
    num_spatial_dims = len(inputs.shape) - 2
    batch_size, _, *image_size = inputs.shape

    # compute all slices
    scan_interval = get_scan_interval_(image_size, roi_size, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range = range(0, total_slices, sw_batch_size)

    return windows_range, num_win, slices, total_slices

@script()
def prepare_for_loop_script(inputs: FLOAT["N", "C", "D", "H", "W"], roi_size: INT64[3], sw_batch_size: INT64=1, overlap: FLOAT=0.25) -> (INT64[2], INT64, INT64["N", 3, 2], INT64):
    input_shape = op.Shape(inputs)
    # batch_size, _, image_size = op.Split(input_shape, [1, 1, 3])
    batch_size, channels, D, H, W = op.Split(input_shape, num_outputs=5)

    image_size_seq = op.SequenceConstruct(D, H, W)
    image_size = op.ConcatFromSequence(image_size_seq, axis=-1, new_axis=0)

    # compute all slices
    scan_interval = get_scan_interval_script(image_size, roi_size, overlap)
    slices = dense_patch_slices_script(image_size, roi_size, scan_interval)

    # num_win, three, two = op.Split(op.Shape(slices), [1, 1, 1])
    num_win, three, two = op.Split(op.Shape(slices), num_outputs=3)
    total_slices = num_win * batch_size  # total number of windows
    windows_range = op.Range(0, total_slices, sw_batch_size)

    return windows_range, num_win, slices, total_slices

# @script()
def prepare_for_predictor(inputs, slice_g, num_win, slices, total_slices, sw_batch_size):
    slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
    unravel_slice = [
        [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
        for idx in slice_range
    ]
    if sw_batch_size > 1:
        win_data = torch.cat([inputs[tuple(win_slice)] for win_slice in unravel_slice])
    else:
        win_data = inputs[tuple(unravel_slice[0])]
    return win_data, unravel_slice

@script()
def prepare_for_predictor_batch_size_is_1_script(inputs: FLOAT[1, 1, "D", "H", "W"], slice_g: INT64, slices: INT64["N", 3, 2], sw_batch_size: INT64=1) -> (FLOAT[1, 1, "D1", "H1", "W1"], INT64[3], INT64[3]):
    # assume batch size N is 1
    num_win, three, two = op.Split(op.Shape(slices), num_outputs=3)
    spatial_axes = op.Range(2, 5, 1)
    zero_int = op.Constant(value_ints=[0])
    one_int = op.Constant(value_ints=[1])
    two_int = op.Constant(value_ints=[2])
    slice_g_ = slice_g + zero_int
    slice_g_1 = slice_g + one_int
    start_ = op.Slice(slices, op.Concat(slice_g_, zero_int, axis=0), op.Concat(slice_g_1, one_int, axis=0), op.Constant(value_ints=[0, 2]))
    start = op.Reshape(start_, op.Constant(value_ints=[-1]))
    stop_ = op.Slice(slices, op.Concat(slice_g_, one_int, axis=0), op.Concat(slice_g_1, two_int, axis=0), op.Constant(value_ints=[0, 2]))
    stop = op.Reshape(stop_, op.Constant(value_ints=[-1]))
    win_data = op.Slice(inputs, start, stop, spatial_axes)
    # if sw_batch_size > 1:
    #     for idx in op.Range(slice_g + 1, op.Min(slice_g + sw_batch_size, num_win)):
    #         win_data = op.Concat(win_data, op.Slice(inputs, slices[idx, :, 0], slices[idx, :, 1], spatial_axes))
    return win_data, start, stop

def process_predictor_output_seg_tuple_length_is_1(seg_tuple, slices, unravel_slice, importance_map, image_size, batch_size):
    # seg_shape and roi_size are of the same shape so no inpolation of weight map is needed.    
    # seg_tuple is a tuple of tensors, each tensor is of shape [seg_chns, *roi_size]
    assert (len(seg_tuple) == 1)
    b_shape = seg_tuple.shape
    seg_chns, seg_shape = b_shape[1], b_shape[2:]
    output_shape = [batch_size, seg_chns]
    output_shape += list(image_size)
    new_tensor: Callable = torch.zeros  # type: ignore
    output_image = new_tensor(output_shape)
    count_map = torch.zeros([1, 1] + output_shape[2:])
    for __s in slices:
        count_map[(slice(None), slice(None), *__s)] += importance_map

    seg_tuple *= importance_map
    for original_idx, p in zip(unravel_slice, seg_tuple):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        output_image[idx_zm] += p

    return output_image, count_map

@script()
def roi_indices_3d(start: INT64[3], stop: INT64[3], step: INT64[3]) -> (INT64["roi_D", "roi_H", "roi_W"], INT64[3]):
    """
    This function is used to generate the indices for a 3d roi window.
    """
    start_d, start_h, start_w = op.Split(start, num_outputs=3)
    stop_d, stop_h, stop_w = op.Split(stop, num_outputs=3)
    step_d, step_h, step_w = op.Split(step, num_outputs=3)
    
    grid_d_0 = op.Range(start_d, stop_d, step_d)
    grid_h_0 = op.Range(start_h, stop_h, step_h)
    grid_w_0 = op.Range(start_w, stop_w, step_w)

    D = op.Shape(grid_d_0)
    H = op.Shape(grid_h_0)
    W = op.Shape(grid_w_0)

    roi_shape = op.Concat(D, H, W, axis=0)
    zeros_D_H_W = op.CastLike(op.ConstantOfShape(roi_shape), start)

    zeros_H_W_D = op.Transpose(zeros_D_H_W, perm=[1, 2, 0])
    grid_d_1 = zeros_H_W_D + grid_d_0
    grid_d = op.Transpose(grid_d_1, perm=[2, 0, 1])

    zeros_D_W_H = op.Transpose(zeros_D_H_W, perm=[0, 2, 1])
    grid_h_1 = zeros_D_W_H + grid_h_0
    grid_h = op.Transpose(grid_h_1, perm=[0, 2, 1])

    grid_w = grid_w_0 + zeros_D_H_W

    indices_seq = op.SequenceConstruct(grid_d, grid_h, grid_w)

    indices = op.ConcatFromSequence(indices_seq, axis=-1, new_axis=1)  # [D, H, W, 3]

    return indices, roi_shape

@script()
def process_predictor_output_seg_tuple_length_is_1_script(
        seg_tuple: FLOAT[1, "seg_C", "roi_D", "roi_H", "roi_W"],
        start: INT64[3],
        stop: INT64[3],
        importance_map: FLOAT["roi_D", "roi_H", "roi_W"],
        aggrregated_pred: FLOAT[1, "seg_C", "D", "H", "W"],
        aggrregated_count: INT64[1, 1, "D", "H", "W"],
        ) -> (FLOAT[1, "seg_C", "D", "H", "W"], FLOAT[1, 1, "D", "H", "W"]):
    """
    This function is used to aggregate the predictor output to the final output, count is used to record the number of
    pixels that are aggregated to the final output.
    """
    # for ScatterND:
    # r = 2 + 3,
    # q = r + 1,
    # k = indices.shape[q - 1] = r
    # indices = [0, 0:"seg_C", start_z:stop_z, start_y:stop_y, start_x:stop_x
    
    step_ones = op.Constant(value_ints=[1, 1, 1])
    indices, roi_shape = roi_indices_3d(start, stop, step_ones)

    weighted_pred = seg_tuple * importance_map
    aggrregated_pred_out = op.ScatterND(aggrregated_pred, indices, weighted_pred, reduction="add")

    count_ones = op.ConstantOfShape(roi_shape, value=onnx.helper.make_tensor("one", TensorProto.INT64, [1], [1]))
    aggrregated_count_out = op.ScatterND(aggrregated_count, indices, count_ones, reduction="add")
    return aggrregated_pred_out, aggrregated_count_out

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
        slices_nd = slices_as_ndarray(slices)
        cases = [
            onnx_script_test_case.FunctionTestParams(dense_patch_slices_script, [image_size, roi_size, np.asarray(interval, dtype=np.int64)], [slices_nd])
        ]
        for case in cases:
            self.run_converter_test(case)
            self.run_eager_test(case)

    def test_prepare_for_loop(self):
        # prepare_for_loop(_script) calls get_scan_interval(_script) and dense_patch_slices(_script) which are individually tested above
        # now we test prepare_for_loop(_script) as a whole
        image_size = np.array([10, 10, 10], dtype=np.int64)
        roi_size = np.array([5, 5, 5], dtype=np.int64)
        overlap = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        inputs = np.random.rand(1, 1, *image_size).astype(np.float32)
        sw_batch_size = 1
        windows_range, num_win, slices, total_slices = prepare_for_loop(inputs, roi_size, sw_batch_size=sw_batch_size, overlap=overlap)
        windows_range_nd = np.asarray(windows_range, dtype=np.int64)
        num_win_nd = np.array([num_win], dtype=np.int64)
        total_slices_nd = np.array([total_slices], dtype=np.int64)
        slices_nd = slices_as_ndarray(slices)
        cases = [
            onnx_script_test_case.FunctionTestParams(prepare_for_loop_script, [inputs, roi_size, np.array([sw_batch_size], dtype=np.int64), overlap], [windows_range_nd, num_win_nd, slices_nd, total_slices_nd])
        ]
        for case in cases:
            # run_converter_test failed at onnxruntime_pybind_mlvalue.cc:110 : Invalid argument
            self.run_converter_test(case)
            # run_eager_test almost passes, but the output a tuple of tensors. need fix np.testing.assert_allclose(...) to handle tuple of tensors
            self.run_eager_test(case)

    def test_in_loop_pre_post_process_for_predictor(self):
        image_size = np.array([128, 128, 128], dtype=np.int64)
        roi_size = np.array([64, 64, 32], dtype=np.int64)
        overlap = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        batch_size = 1
        inputs = np.random.rand(batch_size, 1, *image_size).astype(np.float32)
        sw_batch_size = 1
        windows_range, num_win, slices, total_slices = prepare_for_loop(inputs, roi_size, sw_batch_size=sw_batch_size, overlap=overlap)
        slice_g = 0
        win_data, unravel_slice = prepare_for_predictor(inputs, slice_g, num_win, slices, total_slices, sw_batch_size)
        slice_g_nd = np.array([slice_g], dtype=np.int64)
        sw_batch_size_nd = np.array([sw_batch_size], dtype=np.int64)
        slices_nd = slices_as_ndarray(slices)

        start = np.array([s.start for s in unravel_slice[slice_g][2:]], dtype=np.int64)
        stop = np.array([s.stop for s in unravel_slice[slice_g][2:]], dtype=np.int64)
        cases = [
            onnx_script_test_case.FunctionTestParams(prepare_for_predictor_batch_size_is_1_script, [inputs, slice_g_nd, slices_nd, sw_batch_size_nd], [win_data, start, stop])
        ]
        for case in cases:
            self.run_converter_test(case)
            self.run_eager_test(case)

        config_file = os.path.join(os.path.dirname(__file__), "testing_data", "inference.json")
        inferer = make_test_config_workflow_inferer(config_file)
        inferer.initialize()
        predictor = deepcopy(inferer.network_def)
        predictor.eval()

        with torch.no_grad():
            seg_prob_out = predictor(torch.from_numpy(win_data))
            print(seg_prob_out.shape)

        roi_weight_map = compute_importance_map(tuple(roi_size), mode=BlendMode.CONSTANT, sigma_scale=0.125, device="cpu")
        output_image, count_map = process_predictor_output_seg_tuple_length_is_1(seg_prob_out, slices, unravel_slice, roi_weight_map,image_size, batch_size)
        print(output_image.shape)
        print(count_map.shape)
        
        aggrregated_pred = np.zeros((batch_size, 2, *image_size), dtype=np.float32)
        aggrregated_count = np.zeros((batch_size, 1, *image_size), dtype=np.int64)

        case = onnx_script_test_case.FunctionTestParams(
            process_predictor_output_seg_tuple_length_is_1_script,
            [seg_prob_out, slices, slice_g, roi_weight_map, aggrregated_pred, aggrregated_count],
            [output_image, count_map])
        self.run_converter_test(case)
        self.run_eager_test(case)        


if __name__ == "__main__":
    unittest.main()
