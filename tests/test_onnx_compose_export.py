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

from monai.utils import (BlendMode, PytorchPadMode)


from onnxscript import script
from onnxscript.onnx_opset import opset15 as op
from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.tests.common import onnx_script_test_case, testutils

from onnxscript.tests.models.pnp import roi_indices_3d, aggrregate_predictor_output, sliding_window_inference, predict_mock, predict_mock_2, Opset18Ext


TEST_CASE_1 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.json")]


TEST_CASE_2 = ["C:/LiqunWA/MONAI/model-zoo-fork/models/brats_mri_segmentation/configs/inference.json"]
TEST_CASE_3 = ["C:/LiqunWA/MONAI/bundles/lung_nodule_ct_detection/configs/inference.json"]
TEST_CASE_4 = ["C:/LiqunWA/MONAI/model-zoo-fork/models/spleen_ct_segmentation/configs/inference.json"]
TEST_CASE_5 = ["C:/LiqunWA/MONAI/model-zoo-fork/models/wholeBody_ct_segmentation/configs/inference.json"]

# TEST_CASE_3 = [os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json")]

import torch
import onnx
import onnxruntime as ort
import io

from onnx import numpy_helper, TensorProto

def parse_config_path_get_task_name(config_path):
    path_components = config_path.split("/")

    # Find the index of "model-zoo" or "model-zoo-fork"
    model_zoo_index = next((i for i, component in enumerate(path_components) if component == 'model-zoo' or component == 'model-zoo-fork'), None)

    if model_zoo_index is not None and model_zoo_index + 2 < len(path_components):
        # Extract the subfolder name after "model-zoo" or "model-zoo-fork"
        return path_components[model_zoo_index + 2]
    else:
        return "default"

def make_sliding_window_inferer_model(predictor, sliding_window_inferer, input_dtype, opset_version):
    tensor_type_proto = onnx.helper.make_tensor_type_proto(input_dtype, None)
    node = onnx.helper.make_node(
        "SlidingWindowInferer",
        inputs=["image"],
        outputs=["pred"],
        predictor=predictor.graph,
        roi_size=sliding_window_inferer.roi_size,
        sw_batch_size=sliding_window_inferer.sw_batch_size,
        overlap=sliding_window_inferer.overlap,
        # mode=sliding_window_inferer,
        # sigma_scale=sliding_window_inferer,
        # padding_mode=sliding_window_inferer,
        # cval=sliding_window_inferer,
        # roi_weight_map=sliding_window_inferer
        # roi_size=roi_size,
        # sw_batch_size=sw_batch_size,
        # overlap=overlap
    )
    graph = onnx.helper.make_graph(
        [node],
        "sliding_window_inferer",
        [onnx.helper.make_value_info(name="image", type_proto=tensor_type_proto)],
        [onnx.helper.make_value_info(name="pred", type_proto=tensor_type_proto)])
    model = onnx.helper.make_model(graph, producer_name="MONAI", opset_imports=[onnx.helper.make_opsetid("", opset_version)])
    return model

def save_onnx_test_case(model, inputs, outputs, output_dir):
    def prepare_dir(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    prepare_dir(output_dir)
    with open(os.path.join(output_dir, "model.onnx"), "wb") as f:
        f.write(model.SerializeToString())
    data_set_dir = os.path.join(output_dir, f"test_data_set_{0}")
    prepare_dir(data_set_dir)
    for j, input in enumerate(inputs):
        with open(os.path.join(data_set_dir, f"input_{j}.pb"), "wb") as f:
            if model.graph.input[j].type.HasField("map_type"):
                f.write(
                    numpy_helper.from_dict(
                        input, model.graph.input[j].name
                    ).SerializeToString()
                )
            elif model.graph.input[j].type.HasField("sequence_type"):
                f.write(
                    numpy_helper.from_list(
                        input, model.graph.input[j].name
                    ).SerializeToString()
                )
            elif model.graph.input[j].type.HasField("optional_type"):
                f.write(
                    numpy_helper.from_optional(
                        input, model.graph.input[j].name
                    ).SerializeToString()
                )
            else:
                assert model.graph.input[j].type.HasField(
                    "tensor_type"
                )
                if isinstance(input, TensorProto):
                    f.write(input.SerializeToString())
                else:
                    f.write(
                        numpy_helper.from_array(
                            input, model.graph.input[j].name
                        ).SerializeToString()
                    )
    for j, output in enumerate(outputs):
        with open(os.path.join(data_set_dir, f"output_{j}.pb"), "wb") as f:
            if model.graph.output[j].type.HasField("map_type"):
                f.write(
                    numpy_helper.from_dict(
                        output, model.graph.output[j].name
                    ).SerializeToString()
                )
            elif model.graph.output[j].type.HasField("sequence_type"):
                f.write(
                    numpy_helper.from_list(
                        output, model.graph.output[j].name
                    ).SerializeToString()
                )
            elif model.graph.output[j].type.HasField("optional_type"):
                f.write(
                    numpy_helper.from_optional(
                        output, model.graph.output[j].name
                    ).SerializeToString()
                )
            else:
                assert model.graph.output[j].type.HasField(
                    "tensor_type"
                )
                if isinstance(output, TensorProto):
                    f.write(output.SerializeToString())
                else:
                    f.write(
                        numpy_helper.from_array(
                            output, model.graph.output[j].name
                        ).SerializeToString()
                    )

class ComposeWrapper(torch.nn.Module):
    def __init__(self, compose, image_meta_dict, key):
        super().__init__()
        self.compose = compose
        self.image_meta_dict = image_meta_dict
        self.key = key

    def forward(self, x):
        meta_data = {self.key: x, "image_meta_dict": self.image_meta_dict}
        y = self.compose(meta_data)
        return y[self.key]

class SlidingWindowInfererWrapper(torch.nn.Module):
    def __init__(
            self,
            sliding_window_inferer: monai.inferers.inferer.SlidingWindowInferer,
            network):
        super().__init__()
        self.sliding_window_inferer = sliding_window_inferer
        self.network = network

    def forward(self, x):
        return self.sliding_window_inferer(x, self.network)

class TestBundleWorkflowONNX(onnx_script_test_case.OnnxScriptTestCase):
    def setUp(self):
        # there is AffineGrid only in opset 20.
        # ort release only support up to opset 19
        # pytorch current main only support onnx 13.1 which is opset 18
        self.opset_version = 18
        register_custom_op(opset_version=self.opset_version)

    def tearDown(self):
        pass

    def _is_spleen_ct_segmentation(self, config_file):
        return "spleen_ct_segmentation" in config_file

    def _show_image_and_output_tensor(self, input_tensor, output_tensor, output_tensor2=None, show_coronal=False, slice_index=None, show_grid=False,
                                      aspect_ratio=1, enhance=False):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from scipy.ndimage import zoom
        from skimage import exposure

        image = input_tensor.detach().cpu().numpy() if isinstance(input_tensor, torch.Tensor) else input_tensor
        output = output_tensor.detach().cpu().numpy() if isinstance(output_tensor, torch.Tensor) else output_tensor
        output2 = output_tensor2.detach().cpu().numpy() if isinstance(output_tensor2, torch.Tensor) else output_tensor2

        def get_channels(pred):
            channels = pred.shape[-4] if pred is not None and len(pred.shape) > 3 else 1 if pred is not None else 0
            if channels > 4:
                channels = 4
            return channels

        sub_count = get_channels(image) + get_channels(output) + get_channels(output2)

        plt.figure("check", (12, 6))
        def show_image_channel(image, sub, slice_index=None, cmap="viridis"):
            if show_coronal:
                slice_index = slice_index or image.shape[-2] // 2
            else:
                slice_index = slice_index or image.shape[-1] // 2
            for c in range(get_channels(image)):
                ax = plt.subplot(1, sub_count, sub)
                if show_coronal:
                    if len(image.shape) == 3:
                        corrected_image = np.transpose(image[:, slice_index, ::-1]) # image will otherwise be up side down without -1:0
                    elif len(image.shape) == 4:
                        corrected_image = np.transpose(image[c, :, slice_index, ::-1])
                    else:
                        corrected_image = np.transpose(image[0, c, :, slice_index, ::-1])
                    if aspect_ratio != 1:
                        original_height, original_width = corrected_image.shape
                        corrected_height = original_height // aspect_ratio
                        corrected_image = zoom(corrected_image, (corrected_height/original_height, 1.0))
                else:
                    if len(image.shape) == 3:
                        corrected_image = np.transpose(image[:, ::-1, slice_index])
                    elif len(image.shape) == 4:
                        corrected_image = np.transpose(image[c, :, ::-1, slice_index])
                    else:
                        corrected_image = np.transpose(image[0, c, :, ::-1, slice_index])
                if enhance and cmap == "gray":
                    corrected_image = exposure.equalize_hist(corrected_image)
                plt.imshow(corrected_image, cmap=cmap)
                if show_grid:
                    cols, rows = 4, 4
                    grid_size = corrected_image.shape[-2] // rows + 5 # +5 to show that image size is not multiple of roi size 
                    for i in range(1, cols):
                        plt.axvline(i * grid_size, color='r', linewidth=0.5)

                    # Draw the horizontal grid lines
                    for i in range(1, rows):
                        plt.axhline(i * grid_size, color='r', linewidth=0.5)

                    x, y = 2, 2
                    rect = patches.Rectangle((x * grid_size, y * grid_size), grid_size, grid_size,
                                            facecolor='red' if cmap == "viridis" else "yellow", alpha=0.2)
                    ax.add_patch(rect)
                    
                sub += 1
            return sub

        sub = 1
        sub = show_image_channel(image, sub, slice_index, cmap="gray")

        if output is not None:
            sub = show_image_channel(output, sub, slice_index)
        if output2 is not None:
            sub = show_image_channel(output2, sub, slice_index)
        plt.show()

    def _validate_with_ort(self, model, input_data, output_data):
        ort_session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        assert len(ort_session.get_inputs()) == len(input_data)
        ort_inputs = {}
        for ort_input, input in zip(ort_session.get_inputs(), input_data):
            if isinstance(input, torch.Tensor):
                ort_inputs[ort_input.name] = input.detach().cpu().numpy()
            else:
                ort_inputs[ort_input.name] = input
        ort_outs = ort_session.run(None, ort_inputs)
        for (ort_out, output) in zip(ort_outs, output_data):
            np.testing.assert_allclose(ort_out, output.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)

    def _make_test_config_workflow_inferer(self, config_file, override=None):
        override = override if override is not None else {
            "network": "$@network_def.to('cpu')",
            "dataset#_target_": "Dataset",
            # "dataset#data": [{"image": self.filename}],
            # "postprocessing#transforms#2#output_postfix": "seg",
            # "output_dir": self.data_dir,
        }
        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow="infer",
            config_file=config_file,
            logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
            **override,
        )
        return inferer

    def _run_up_to_preprocessing(self, inferer):
        dataset = inferer.parser.get_parsed_content("dataset")

        def collate_fn(dataset):
            images = []
            image_metas = []
            for meta in dataset:
                images = [meta["image"], *images]
                metas = [meta["image_meta_dict"], *image_metas]
            return {"image": torch.stack(images), "image_meta_dict": metas}

        input_image_and_image_meta_dict_pair = collate_fn(dataset)
        return input_image_and_image_meta_dict_pair

    def _run_up_to_sliding_window(self, inferer):
        input_image_and_image_meta_dict_pair = self._run_up_to_preprocessing(inferer)

        sliding_window = inferer.inferer
        self.assertTrue(isinstance(sliding_window, SlidingWindowInferer))

        pred = sliding_window(input_image_and_image_meta_dict_pair["image"], inferer.network_def)
        return {"pred": pred, "image_meta_dict": input_image_and_image_meta_dict_pair["image_meta_dict"]}

    def _run_sliding_window_inferer(self, inferer, input_image_and_image_meta_dict_pair):
        sliding_window = inferer.inferer
        self.assertTrue(isinstance(sliding_window, SlidingWindowInferer))

        pred = sliding_window(input_image_and_image_meta_dict_pair["image"], inferer.network_def)
        return {"pred": pred, "image_meta_dict": input_image_and_image_meta_dict_pair["image_meta_dict"]}

    def _run_postprocessing(self, postprocessing, pred_and_image_meta_dict):
        postprocessed_image_and_image_meta_dict_pair = None
        for i, process in enumerate(postprocessing.transforms):
            if isinstance(process, monai.transforms.io.dictionary.SaveImaged):
                break
            if postprocessed_image_and_image_meta_dict_pair is None:
                postprocessed_image_and_image_meta_dict_pair = process(pred_and_image_meta_dict)
            else:
                postprocessed_image_and_image_meta_dict_pair = process(postprocessed_image_and_image_meta_dict_pair)
        return postprocessed_image_and_image_meta_dict_pair["pred"]

    def export_compose(pnp_compose, opset_version, input_tensor, output_tensor, image_meta_dict, task_name):
        compose_wrapper = ComposeWrapper(pnp_compose, image_meta_dict, "image")

        f = io.BytesIO()
        torch.onnx.export(compose_wrapper, input_tensor, f, opset_version=opset_version)

        onnx_model = onnx.load_model_from_string(f.getvalue())

        onnx.save(onnx_model, f"c:/temp/monai_{task_name}_preprocessing_compose.onnx")

        try:
            validate_with_ort(onnx_model, [input_tensor], [output_tensor])
            save_onnx_test_case(onnx_model, [input_tensor.detach().numpy()], [output_tensor.detach().numpy()], f"c:/temp/monai_{task_name}_preprocessing_compose")
        except Exception as e:
            # FAIL : Fatal error: custom:AffineGrid(-1) is not a registered function/op
            print(f"Failed to validate {task_name} with ort: {e}")

    @parameterized.expand([
        # TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        TEST_CASE_4,
        # TEST_CASE_5,
        ])
    def test_convert_preprocess_compose_to_onnx(self, config_file):
        inferer = self._make_test_config_workflow_inferer(config_file)

        inferer.initialize()
        # dataset_datalist = inferer.parser.get_parsed_content("dataset")
        # dataset_data0 = dataset_datalist[0]       # "data": "@_meta_#datalist",
        dataset = inferer.parser.get_parsed_content("dataset")       # "data": "@_meta_#datalist",
        preprocessing = inferer.preprocessing
        load_imaged = preprocessing.transforms[0]
        preprocessing.transforms = preprocessing.transforms[1:]

        input_image_and_image_meta_dict_pair = load_imaged(dataset.data[0])
        processed_image_and_image_meta_dict_pair = None
        for p in preprocessing.transforms:
            if processed_image_and_image_meta_dict_pair is None:
                processed_image_and_image_meta_dict_pair = p(input_image_and_image_meta_dict_pair)
            else:
                processed_image_and_image_meta_dict_pair = p(processed_image_and_image_meta_dict_pair)

        image_meta_dict = input_image_and_image_meta_dict_pair["image_meta_dict"]
        input_meta_tensor = input_image_and_image_meta_dict_pair["image"]
        input_tensor = input_meta_tensor.as_tensor()
        processed_output_metatensor = processed_image_and_image_meta_dict_pair["image"]
        output_tensor = processed_output_metatensor.as_tensor()

        self._show_image_and_output_tensor(input_meta_tensor, processed_output_metatensor,)

        compose_wrapper = ComposeWrapper(preprocessing, image_meta_dict, "image")

        f = io.BytesIO()
        if self.opset_version == 20:
            # I need to add onnx_shape_inference to onnx.export api and set it to False to skip
            #if shape_inference:
            #   _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
            # otherwise, it will fail with: RuntimeError: Unsupported onnx_opset_version: 20
            # however, I am still getting the following error:
            #
            torch.onnx.export(compose_wrapper, input_tensor, f, opset_version=self.opset_version, onnx_shape_inference=False)
        else:
            # TEST_CASE_5: ValueError: Unknown original_channel_dim in the MetaTensor meta dict or `meta_dict` or `channel_dim`.
            torch.onnx.export(compose_wrapper, input_tensor, f, opset_version=self.opset_version)

        onnx_model = onnx.load_model_from_string(f.getvalue())

        task_name = parse_config_path_get_task_name(config_file)
        onnx.save(onnx_model, f"c:/temp/monai_{task_name}_preprocessing_compose.onnx")

        try:
            self._validate_with_ort(onnx_model, [input_tensor], [output_tensor])
            save_onnx_test_case(onnx_model, [input_tensor.detach().numpy()], [output_tensor.detach().numpy()], f"c:/temp/monai_{task_name}_preprocessing_compose")
        except Exception as e:
            # FAIL : Fatal error: custom:AffineGrid(-1) is not a registered function/op
            print(f"Failed to validate {config_file} with ort: {e}")

    @parameterized.expand([
        # TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        # TEST_CASE_4,
        TEST_CASE_5,
        ])
    def test_convert_postprocess_compose_to_onnx(self, config_file):
        inferer = self._make_test_config_workflow_inferer(config_file)
        inferer.initialize()
        # skip postprocessing in the workflow
        postprocessing = inferer.postprocessing
        inferer.postprocessing = None

        # should initialize and parse again as changed the bundle content
        inferer.initialize()
        inferer.run()
        inferer.finalize()

        aspect_ratio_tensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.batch[0]["image_meta_dict"]["pixdim"][1] / inferer.parser.ref_resolver.resolved_content["evaluator"].state.batch[0]["image_meta_dict"]["pixdim"][0]
        aspect_ratio = aspect_ratio_tensor.item()

        # show input image and predictor output
        input_image_metatensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["image"]
        pred_metatensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"]

        # for ONNX presentation, show input in coronal view, no grid, correct aspect_ratio
        self._show_image_and_output_tensor(input_image_metatensor, pred_metatensor, show_coronal=True, show_grid=False, aspect_ratio=aspect_ratio)
        # for ONNX presentation, axial view, no grid, no need to correct aspect_ratio
        self._show_image_and_output_tensor(input_image_metatensor, pred_metatensor, show_coronal=False, show_grid=False)
        # for ONNX presentation, show input and pred with sliding window gird in axial view, enhance the image
        self._show_image_and_output_tensor(input_image_metatensor, pred_metatensor, show_coronal=False, show_grid=True, enhance=False)

        # prepare and run postprocessing
        pred_and_image_meta_dict = {
            "pred": inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"],
            "image_meta_dict": inferer.parser.ref_resolver.resolved_content["evaluator"].state.batch[0]["image_meta_dict"]}
        post_processed_metatensor = self._run_postprocessing(postprocessing, pred_and_image_meta_dict)

        # for ONNX presentation, show segmentation in axial view
        self._show_image_and_output_tensor(input_image_metatensor, post_processed_metatensor, show_coronal=False, show_grid=False)
        # for ONNX presentation, show segmentation in coronal view
        self._show_image_and_output_tensor(input_image_metatensor, post_processed_metatensor, show_coronal=True, show_grid=False, aspect_ratio=aspect_ratio)

        image_meta_dict = inferer.parser.ref_resolver.resolved_content["evaluator"].state.batch[0]["image_meta_dict"]
        compose_wrapper = ComposeWrapper(postprocessing, image_meta_dict, "pred")

        f = io.BytesIO()
        input_tensor = pred_metatensor.as_tensor()
        post_processed_tensor = post_processed_metatensor.as_tensor()
        torch.onnx.export(compose_wrapper, input_tensor, f, opset_version=self.opset_version)
        onnx_model = onnx.load_model_from_string(f.getvalue())

        task_name = parse_config_path_get_task_name(config_file)
        onnx.save(onnx_model, f"c:/temp/monai_{task_name}_postprocessing_compose.onnx")

        try:
            self._validate_with_ort(onnx_model, [input_tensor], [post_processed_tensor])
            save_onnx_test_case(onnx_model, [input_tensor.detach().numpy()], [post_processed_tensor.detach().numpy()], f"c:/temp/monai_{task_name}_postprocessing_compose")
        except Exception as e:
            # with TEST_CASE_5, due to size of sample input/output data tensor: Message onnx.TensorProto exceeds maximum protobuf size of 2GB: 5833780289
            print(f"Failed to validate {config_file} with ort: {e}")


    @parameterized.expand([
        # TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        TEST_CASE_4,
        ])
    def test_sliding_window_inferer2_and_inferer_same_output(self, config_file):
        # dev test to make sure SlidingWindowInferer2 produces the same output as SlidingWindowInferer
        # SlidingWindowInferer2 is being refactored to invoke several methods that can be separately exported to ONNX.
        inferer = self._make_test_config_workflow_inferer(config_file)
        # inferer.bundle_root = "C:/LiqunWA/MONAI/model-zoo-fork"
        inferer.initialize()
        predictor = inferer.network_def
        inferer.network_def = deepcopy(predictor)
        inferer.initialize()
        sliding_window_inferer = inferer.inferer
        inferer.inferer = deepcopy(sliding_window_inferer)
        inferer.initialize()

        postprocessing = inferer.postprocessing
        inferer.postprocessing = None
        # should initialize and parse again as changed the bundle content
        inferer.initialize()

        inferer.run()
        inferer.finalize()

        image_meta_tensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["image"]
        pred_meta_tensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"]
        image_meta_tensor = torch.unsqueeze(image_meta_tensor, 0)
        pred_meta_tensor = torch.unsqueeze(pred_meta_tensor, 0)

        self._show_image_and_output_tensor(image_meta_tensor, pred_meta_tensor,)

        print("run sliding window with model.pt")
        predictor.load_state_dict(torch.load("C:/LiqunWA/MONAI/model-zoo-fork/models/spleen_ct_segmentation/models/model.pt"))
        sw2 = SlidingWindowInferer2(
            predictor, sliding_window_inferer.roi_size, sliding_window_inferer.sw_batch_size,
            sliding_window_inferer.overlap,
            )
        actual_output_sw2 = sw2(image_meta_tensor)
        sw = SlidingWindowInferer(
            sliding_window_inferer.roi_size, sliding_window_inferer.sw_batch_size,
            sliding_window_inferer.overlap,
            )
        actual_output_sw = sw(image_meta_tensor, predictor)
        self._show_image_and_output_tensor(image_meta_tensor, actual_output_sw, actual_output_sw2,)
        self.assertTrue(torch.allclose(actual_output_sw, actual_output_sw2))

        # override sw_batch_size to 1 and overlap to 0
        sw_batch_size_override = 1
        overlap_override = 0
        roi_size_override = (192, 192, 128)
        sw2_o = SlidingWindowInferer2(
            predictor, roi_size_override, sw_batch_size_override,
            overlap_override,
            )
        actual_output_sw2_o = sw2_o(image_meta_tensor)
        sw_o = SlidingWindowInferer(
            roi_size_override, sw_batch_size_override,
            overlap_override,
            )
        actual_output_sw_o = sw_o(image_meta_tensor, predictor)
        self._show_image_and_output_tensor(image_meta_tensor, actual_output_sw_o, actual_output_sw2_o,)
        self.assertTrue(torch.allclose(actual_output_sw_o, actual_output_sw2_o))




    @parameterized.expand([
        TEST_CASE_5,
        ])
    def test_sliding_window_inferer_whole_body_sliding_along_z_axis(self, config_file):
        # input data shape torch.Size([1, 253, 253, 217]). use roi so that it only slide along z axis
        override = {
            "inferer#roi_size": [256, 256, 96],
        }
        inferer = self._make_test_config_workflow_inferer(config_file, override=override)
        # inferer = self._make_test_config_workflow_inferer(config_file)
        inferer.initialize()
        inferer.run()
        inferer.finalize()

        self._show_image_and_output_tensor(
            inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["image"],
            inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"],)

        print("completed")

    @parameterized.expand([
        TEST_CASE_4,
        ])
    def test_sliding_window_inferer_simple(self, config_file):
        # input shape: torch.Size([1, 1, 220, 220, 84])
        # for simplicity, make step size the same as roi size (overlay = 0), let sw_batch_size = 1. No padding (roi_size <= input_size)
        # sliding_window_inference uses pred = op.OpaqueOp(win_data, model_path="C:/Temp/sliding_window_predictor_sw_batch_size_is_1.onnx")
        # the onnx model has fixed input size (64, 64, 32).
        # TODO: create a model with dynamic input size
        override = {
            "inferer#roi_size": [64, 64, 32],
            "inferer#overlap": 0.0,
            "inferer#sw_batch_size": 1,
        }
        inferer = self._make_test_config_workflow_inferer(config_file, override=override)
        inferer.initialize()
        predictor = inferer.network_def
        inferer.network_def = deepcopy(predictor)
        inferer.initialize()
        inferer.postprocessing = None
        inferer.initialize()

        inferer.run()
        inferer.finalize()

        predictor_input_meta_tensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["image"]
        predictor_output_meta_tensor = inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"]
        self._show_image_and_output_tensor(predictor_input_meta_tensor, predictor_output_meta_tensor,)

        sw_o = SlidingWindowInferer(
            override["inferer#roi_size"], override["inferer#sw_batch_size"],
            override["inferer#overlap"],
            )
        actual_output_sw_o = sw_o(predictor_input_meta_tensor, predictor)

        input_np_array = predictor_input_meta_tensor.image = predictor_input_meta_tensor.detach().cpu().numpy()
        if len(input_np_array.shape) == 4:
            input_np_array = np.expand_dims(input_np_array, 0)

        roi_size_np_array = np.array(override["inferer#roi_size"], dtype=np.int64)
        output_expected_np_array = predictor_output_meta_tensor.detach().cpu().numpy()
        case = onnx_script_test_case.FunctionTestParams(
            sliding_window_inference,
            [input_np_array, roi_size_np_array],
            [output_expected_np_array],
            )
        self.run_eager_test(case)

        print("completed")

    def test_sliding_window_inference(self):
        N, C, D, H, W = 1, 1, 100, 111, 127
        roi_D, roi_H, roi_W = 64, 64, 32
        input = np.ones((N, C, D, H, W), dtype=np.float32)
        roi_size = np.array([roi_D, roi_H, roi_W], dtype=np.int64)

        #output = predict_mock_2(input)
        seg_C = 2
        output_expected = np.zeros((N, seg_C, D, H, W), dtype=np.float32)
        outout_count = np.zeros((N, 1, D, H, W), dtype=np.int64)
        op = Opset18Ext()
        for d in range(0, D, roi_D):
            if d + roi_D > D:
                d = D - roi_D
            for h in range(0, H, roi_H):
                if h + roi_H > H:
                    h = H - roi_H
                for w in range(0, W, roi_W):
                    if w + roi_W > W:
                        w = W - roi_W
                    input_patch = input[:, :, d:d+roi_D, h:h+roi_H, w:w+roi_W]
                    output_expected[:, :, d:d+roi_D, h:h+roi_H, w:w+roi_W] += op.OpaqueOp(input_patch, model_path="C:/Temp/sliding_window_predictor_sw_batch_size_is_1.onnx")
                    outout_count[:, :, d:d+roi_D, h:h+roi_H, w:w+roi_W] += 1

        output_expected /= outout_count

        save_model = False
        if save_model:
            model = sliding_window_inference.function_ir.to_model_proto(producer_name="monai")
            onnx.save(model, "C:/temp/test_sliding_window_inference.onnx")
        case = onnx_script_test_case.FunctionTestParams(
            sliding_window_inference,
            [input, roi_size],
            [output_expected],
            )
        self.run_eager_test(case)
        try:
            # converter test expect to fail with "No Op registered for OpaqueOp with domain_version of 18"
            self.run_converter_test(case)
        except AssertionError as e:
            assert "Verification of model failed" in str(e)
            if isinstance(e.__cause__, onnx.onnx_cpp2py_export.checker.ValidationError):
                root_exception = e.__cause__
                # Handle the root exception here
                print("Root Exception:", root_exception)
                assert "No Op registered for OpaqueOp with domain_version of 18" in str(root_exception)

    @parameterized.expand([
        # TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        TEST_CASE_4,
        ])
    def test_pad_inputs_if_needed_by_roi_size_script(self, config_file):
        inferer = self._make_test_config_workflow_inferer(config_file)
        inferer.bundle_root = "C:/LiqunWA/MONAI/model-zoo-fork"
        inferer.initialize()
        predictor = deepcopy(inferer.network_def)
        sliding_window_inferer = deepcopy(inferer.inferer)

        task_name = parse_config_path_get_task_name(config_file)

        input_image_and_image_meta_dict_pair = self._run_up_to_preprocessing(inferer)
        inputs = input_image_and_image_meta_dict_pair["image"]
        roi_size = np.asarray(sliding_window_inferer.roi_size)

        class Wrapper(torch.nn.Module):
            def __init__(self,
                        padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
                        cval: float = 0.0,):
                super().__init__()
                self.padding_mode = padding_mode
                self.cval = cval

            def forward(self, inputs, roi_size):
                outputs_, pad_size_ = pad_inputs_if_needed_by_roi_size_script(inputs, roi_size, self.padding_mode, self.cval)
                return outputs_, torch.stack(pad_size_)

        wrapper = Wrapper()
        expected_outputs, expected_pad_size = pad_inputs_if_needed_by_roi_size_script(inputs, roi_size, wrapper.padding_mode, wrapper.cval)
        if any(expected_pad_size):
            # inputs need to be padded
            onnx_model = monai.networks.convert_to_onnx(
                wrapper,
                [inputs.as_tensor(), torch.from_numpy(roi_size)],
                ["inputs", "roi_size"],
                ["outputs", "pad_sizes"],
                opset_version=self.opset_version,
                use_trace=True,)

            onnx.save(onnx_model, f"c:/temp/monai_{task_name}_pad_inputs_if_needed_by_roi_size_script.onnx")
            try:
                self._validate_with_ort(onnx_model, [inputs, roi_size], [expected_outputs, expected_pad_size])
                save_onnx_test_case(onnx_model, [inputs, roi_size], [expected_outputs.detach().numpy(), expected_pad_size.detach().numpy()],
                                    f"c:/temp/monai_{task_name}_pad_inputs_if_needed_by_roi_size_script")
            except Exception as e:
                print(f"Failed to validate {config_file} with ort: {e}")

        class Wrapper2(torch.nn.Module):
            def __init__(self,
                         sw_batch_size: int=1,
                         overlap: float=0.25,
                         mode: BlendMode | str = BlendMode.CONSTANT,
                         sigma_scale: Sequence[float] | float = 0.125,
                         ):
                super().__init__()
                self.sw_batch_size = sw_batch_size
                self.overlap = overlap
                self.mode = mode
                self.sigma_scale = sigma_scale

            def forward(self, inputs, roi_size, roi_weight_map):
                windows_range, importance_map, num_win, total_slices, slices, batch_size, image_size = sw_prepare_for_loop(
                    inputs, roi_size, roi_weight_map, self.sw_batch_size, self.overlap, self.mode, self.sigma_scale)
                # slices are list tuples(3 for 3D or 2 for 2D) of slice(start, stop,)
                # to make tracer happy, we need to return a tensor of shape (N, 3 or 2, 2)
                slices_tensor = slices_to_tensor(slices)
                return (
                    torch.range(windows_range.start, windows_range.stop, windows_range.step),
                    importance_map,
                    torch.tensor(num_win),
                    total_slices,
                    slices_tensor,
                    batch_size,
                    torch.stack(image_size)
                )

        wrapper2 = Wrapper2()
        onnx_model2 = monai.networks.convert_to_onnx(
            wrapper2,
            [expected_outputs.as_tensor(), torch.from_numpy(roi_size), None],
            ["inputs", "roi_size"],
            ["windows_range", "importance_map", "num_win", "total_slices", "slices", "batch_size", "image_size"],
            opset_version=self.opset_version,
            use_trace=True,)

        onnx.save(onnx_model2, f"c:/temp/monai_{task_name}_sw_prepare_for_loop.onnx")
        ex_windows_range, ex_importance_map, ex_num_win, ex_total_slices, ex_slices, ex_batch_size, ex_image_size = sw_prepare_for_loop(
            expected_outputs.as_tensor(), roi_size, None, wrapper2.sw_batch_size, wrapper2.overlap, wrapper2.mode, wrapper2.sigma_scale)

        try:
            self._validate_with_ort(
                onnx_model2,
                [expected_outputs.as_tensor(), roi_size, None, wrapper2.sw_batch_size, wrapper2.overlap, wrapper2.mode, wrapper2.sigma_scale],
                [ex_windows_range, ex_importance_map, ex_num_win, ex_total_slices, ex_slices, ex_batch_size, ex_image_size])
            save_onnx_test_case(
                onnx_model2,
                [expected_outputs.as_tensor(), roi_size, None, wrapper2.sw_batch_size, wrapper2.overlap, wrapper2.mode, wrapper2.sigma_scale],
                [ex_windows_range, ex_importance_map, ex_num_win, ex_total_slices, ex_slices, ex_batch_size, ex_image_size],
                f"c:/temp/monai_{task_name}_sw_prepare_for_loop")
        except Exception as e:
            print(f"Failed to validate {config_file} with ort: {e}")


        for slice_g in ex_windows_range:
            ex_slices_tensor = slices_to_tensor(ex_slices)
            ex_win_data, ex_unravel_slice = sw_prepare_win_data_for_predictor_script(
                expected_outputs, slice_g, ex_num_win, ex_slices_tensor, ex_total_slices, wrapper2.sw_batch_size)
            class Wrapper3(torch.nn.Module):
                def __init__(self,
                            sw_batch_size: int=1,
                            ):
                    super().__init__()
                    self.sw_batch_size = sw_batch_size

                def forward(self, inputs, slice_g, num_win, slices, total_slices):
                    slices_list = tensor_to_slices(slices)
                    ex_win_data, ex_unravel_slice = sw_prepare_win_data_for_predictor_script(
                        inputs, slice_g, num_win, slices_list, total_slices, self.sw_batch_size)
                    ex_unravel_slice_tensor = slices_to_tensor(ex_unravel_slice)
                    return ex_win_data, ex_unravel_slice_tensor

            wrapper3 = Wrapper3()

            onnx_model3 = monai.networks.convert_to_onnx(
                wrapper3,
                [expected_outputs.as_tensor(), torch.tensor(slice_g), torch.tensor(ex_num_win), ex_slices_tensor, torch.tensor(ex_total_slices)],
                ["expected_outputs", "slice_g", "ex_num_win", "ex_slices", "ex_total_slices"],
                ["win_data", "unravel_slice"],
                opset_version=self.opset_version,
                use_trace=True,)

            onnx.save(onnx_model3, f"c:/temp/monai_{task_name}_sw_prepare_win_data_for_predictor_script.onnx")

            ex_win_data, ex_unravel_slice = sw_prepare_win_data_for_predictor_script(inputs, slice_g, ex_num_win, ex_slices, ex_total_slices, wrapper3.sw_batch_size)
            # model has mismatched inputs
            # try:
            #     self._validate_with_ort(onnx_model3, [inputs, slice_g, ex_num_win, ex_slices, ex_total_slices], [ex_win_data, ex_unravel_slice])
            #     save_onnx_test_case(
            #         onnx_model3,
            #         [inputs, slice_g, ex_num_win, ex_slices, ex_total_slices],
            #         [ex_win_data, ex_unravel_slice],
            #         f"c:/temp/monai_{task_name}_sw_prepare_win_data_for_predictor_script")
            # except Exception as e:
            #     print(f"Failed to validate {config_file} with ort: {e}")
            break


        class Wrapper4(torch.nn.Module):
            def __init__(self,
                        sw_batch_size: int=1,
                        ):
                super().__init__()
                self.sw_batch_size = sw_batch_size

            def forward(self, inputs, slice_g, num_win, slices, total_slices):
                slices_list = tensor_to_slices(slices)
                ex_win_data, ex_unravel_slice = sw_prepare_win_data_for_predictor_script(
                    inputs, slice_g, num_win, slices_list, total_slices, self.sw_batch_size)
                ex_unravel_slice_tensor = slices_to_tensor(ex_unravel_slice)
                return ex_win_data, ex_unravel_slice_tensor


    @parameterized.expand([
        TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        # TEST_CASE_4,
        ])
    def test_convert_sliding_window_inferer_to_onnx(self, config_file):
        inferer = self._make_test_config_workflow_inferer(config_file)
        input_image_and_image_meta_dict_pair = self._run_up_to_preprocessing(inferer)
        output_pred_and_image_meta_dict_pair = self._run_sliding_window_inferer(inferer, input_image_and_image_meta_dict_pair)
        output_pred_meta_tensor = output_pred_and_image_meta_dict_pair["pred"]

        input_image_tensor = input_image_and_image_meta_dict_pair["image"].as_tensor()

        network = deepcopy(inferer.network_def)
        sliding_window_inferer = deepcopy(inferer.inferer)

        use_trace = False
        if use_trace:
            sw_wrapper = SlidingWindowInfererWrapper(sliding_window_inferer, network)
        if not use_trace:
            sw_wrapper = SlidingWindowInferer2(
                network, sliding_window_inferer.roi_size, sliding_window_inferer.sw_batch_size, sliding_window_inferer.overlap)
            sw_wrapper = torch.jit.script(sw_wrapper)
            sw_wrapper.sliding_window_inferer = sliding_window_inferer
            sw_wrapper.network = network

        f = io.BytesIO()
        torch.onnx.export(
            sw_wrapper,
            input_image_tensor,
            f,
            opset_version=self.opset_version,
            input_names=["image"],
            output_names=["pred"])
        onnx_model = onnx.load_model_from_string(f.getvalue())
        if use_trace:
            onnx.save(onnx_model, "c:/temp/sliding_window_inferer_trace.onnx")
        else:
            onnx.save(onnx_model, "c:/temp/sliding_window_inferer_script.onnx")

    @parameterized.expand([
        TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        # TEST_CASE_4,
        ])
    def test_convert_sw_inferer_to_onnx_sliding_window_node(self, config_file):
        inferer = self._make_test_config_workflow_inferer(config_file)
        input_image_and_image_meta_dict_pair = self._run_up_to_preprocessing(inferer)
        output_pred_and_image_meta_dict_pair = self._run_sliding_window_inferer(inferer, input_image_and_image_meta_dict_pair)
        output_pred_meta_tensor = output_pred_and_image_meta_dict_pair["pred"]

        input_image_tensor = input_image_and_image_meta_dict_pair["image"].as_tensor().numpy()
        network = deepcopy(inferer.network_def)
        sliding_window_inferer = inferer.inferer

        num_input_channel = input_image_tensor.shape[1]
        roi_input_shape = [sliding_window_inferer.sw_batch_size, num_input_channel, *sliding_window_inferer.roi_size]
        roi_input = torch.from_numpy(np.random.randn(*roi_input_shape).astype(input_image_tensor.dtype))
        predictor = monai.networks.convert_to_onnx(
            network,
            inputs=[roi_input],
            input_names=["image_roi"],
            output_names=["pred_roi"],
            opset_version=self.opset_version,
            verify=True,
            use_ort=True,
            atol=1e-5,)

        onnx.save(predictor, "c:/temp/sliding_window_predictor.onnx")
        network.cpu()
        # roi_input = roi_input.to("cuda:0")
        roi_output = network.forward(roi_input)
        save_onnx_test_case(predictor, [roi_input.numpy()], [roi_output.detach().numpy()], "c:/temp/monai_predictor")

        input_dtype = onnx.helper.np_dtype_to_tensor_dtype(input_image_tensor.dtype)
        onnx_model = make_sliding_window_inferer_model(predictor, sliding_window_inferer, input_dtype, self.opset_version)
        onnx.save(onnx_model, "c:/temp/sliding_window_inferer.onnx")

        save_onnx_test_case(onnx_model, [input_image_tensor], [output_pred_meta_tensor.as_tensor().detach().numpy()], "c:/temp/monai_sliding_window_inferer")

    def _test_inferer(self, inferer):
        # should initialize before parsing any bundle content
        inferer.initialize()
        # test required and optional properties
        self.assertListEqual(inferer.check_properties(), [])
        # test read / write the properties, note that we don't assume it as JSON or YAML config here
        # self.assertEqual(inferer.bundle_root, "will override")
        # self.assertEqual(inferer.device, torch.device("cpu"))
        net = inferer.network_def
        self.assertTrue(isinstance(net, UNet))
        sliding_window = inferer.inferer
        self.assertTrue(isinstance(sliding_window, SlidingWindowInferer))
        preprocessing = inferer.preprocessing
        self.assertTrue(isinstance(preprocessing, Compose))
        postprocessing = inferer.postprocessing
        self.assertTrue(isinstance(postprocessing, Compose))
        # test optional properties get
        self.assertTrue(inferer.key_metric is None)
        # inferer.bundle_root = "/workspace/data/spleen_ct_segmentation"
        # inferer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        do_deep_copy = False
        if do_deep_copy:
            inferer.network_def = deepcopy(net)
            inferer.inferer = deepcopy(sliding_window)
            inferer.preprocessing = deepcopy(preprocessing)
        inferer.postprocessing = None
        # test optional properties set
        inferer.key_metric = "set optional properties"

        # should initialize and parse again as changed the bundle content
        inferer.initialize()
        inferer.run()
        inferer.finalize()

        # shpw the image and output tensor which can be predictor output or postprocessing output
        self._show_image_and_output_tensor(
            inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["image"],
            inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"],)

        if inferer.postprocessing is None:
            # just showned the predictor output,
            # process the output with postprocessing and show result
            pred_and_image_meta_dict = {
                "pred": inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["pred"],
                "image_meta_dict": inferer.parser.ref_resolver.resolved_content["evaluator"].state.batch[0]["image_meta_dict"]}
            post_processed = self._run_postprocessing(postprocessing, pred_and_image_meta_dict)

            self._show_image_and_output_tensor(
                inferer.parser.ref_resolver.resolved_content["evaluator"].state.output[0]["image"],
                post_processed,)

        print("")


    @parameterized.expand([
        # TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        TEST_CASE_4,
        ])
    def test_workflow(self, config_file):
        override = {
            # "network": "$@network_def.to(@device)",
            "network": "$@network_def.to('cpu')",
            "dataset#_target_": "Dataset",
            # "dataset#data": [{"image": self.filename}],
            # "postprocessing#transforms#2#output_postfix": "seg",
            "output_dir": "$@bundle_root + '/eval'",
            # "output_dir": self.data_dir,
        }

        # inferer = self._make_test_config_workflow_inferer(config_file)
        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow="infer",
            config_file=config_file,
            logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)

    @parameterized.expand([
        # TEST_CASE_1,
        # TEST_CASE_2,
        # TEST_CASE_3,
        TEST_CASE_4,
        ])
    def test_inference_config_onnx(self, config_file):
        override = {
            "network": "$@network_def.to(@device)",
            "dataset#_target_": "Dataset",
            "dataset#data": [{"image": self.filename}],
            "postprocessing#transforms#2#output_postfix": "seg",
            "output_dir": self.data_dir,
        }
        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow="infer",
            config_file=config_file,
            logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)


if __name__ == "__main__":
    unittest.main()
