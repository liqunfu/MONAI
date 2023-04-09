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

import unittest

from collections.abc import Callable
import io
import onnx
from onnx import helper, TensorProto
import torch
from parameterized import parameterized

from monai.engines import Evaluator, SupervisedEvaluator
from monai.handlers import PostProcessing
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import Activationsd, AsDiscreted, Compose, CopyItemsd, Transform
from tests.utils import assert_allclose

from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import glob
import os


# test lambda function as `transform`
TEST_CASE_1 = [{"transform": lambda x: dict(pred=x["pred"] + 1.0)}, False, torch.tensor([[[[1.9975], [1.9997]]]])]
# test composed postprocessing transforms as `transform`
TEST_CASE_2 = [
    {
        "transform": Compose(
            [
                CopyItemsd(keys="filename", times=1, names="filename_bak"),
                AsDiscreted(keys="pred", threshold=0.5, to_onehot=2),
            ]
        ),
        "event": "iteration_completed",
    },
    True,
    torch.tensor([[[[1.0], [1.0]], [[0.0], [0.0]]]]),
]

class TempModule(torch.nn.Module):
    def __init__(self,
                network: torch.nn.Module | None = None,
                inferer: Inferer | None = None,
                preprocessing: Callable | None = None,
                preprocessing_key: str | None = None,
                postprocessing: Transform | None = None,
                postprocessing_key: str | None = None,
                image_meta_dict: dict | None = None,
                ):
        super().__init__()
        self.network = network
        self.inferer = inferer
        self.preprocessing = preprocessing
        self.preprocessing_key = preprocessing_key
        self.postprocessing = postprocessing
        self.postprocessing_key = postprocessing_key
        self.image_meta_dict = image_meta_dict

    def forward(self, x):
        if self.preprocessing is not None:
            meta_data = {self.preprocessing_key: x, "image_meta_dict": self.image_meta_dict}
            x = self.preprocessing(meta_data)
            x = x[self.preprocessing_key]

        if self.inferer is not None:
            y = self.inferer(x, self.network)
        elif self.network is not None:
            y = self.network(x)
        else:
            y = x

        if self.postprocessing is not None:
            assert self.postprocessing_key is not None and self.image_meta_dict is not None
            meta_data = {self.postprocessing_key: x, "image_meta_dict": self.image_meta_dict}
            y = self.postprocessing(meta_data)
            y = y[self.postprocessing_key]
        return y

class TestEvaluatorToOnnx(unittest.TestCase):
    def test_preprocessing(self):
        device = torch.device("cpu")
        data_dir = "C:/LiqunWA/MONAI/data/Task09_Spleen"
        images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
        data = [{"image": image} for image in images]
        load_and_preprocess_transforms = Compose(
            [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

        load_and_preprocess_transforms_ds = Dataset(data=data, transform=load_and_preprocess_transforms)
        load_and_preprocessed_tensor = load_and_preprocess_transforms_ds[0]["image"].as_tensor().to(device)

        load_transforms = Compose(
            [
            LoadImaged(keys="image"),
            ]
        )
        load_ds = Dataset(data=data, transform=load_transforms)
        loaded_image_tensor = load_ds[0]["image"].as_tensor().to(device)
        image_meta_dict = load_ds[0]["image_meta_dict"]

        preprocess_transforms = Compose(
            [
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

        opset_version = 18
        with torch.no_grad():
            temp_module = TempModule(preprocessing=preprocess_transforms, preprocessing_key="image", image_meta_dict=image_meta_dict)

            def custom_lift_fresh(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
                return g.op("com.microsoft::lift_fresh")

            def affine_grid_generator(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
                # return g.op("com.microsoft::affine_grid_generator")
                theta=torch.tensor([[[9.8824e-01, 0.0000e+00, 0.0000e+00, 5.8824e-03], [0.0000e+00, 1.0030e+00, 0.0000e+00, 4.3355e-04], [0.0000e+00, 0.0000e+00, 1.0030e+00, 4.3355e-04]]], dtype=torch.float64)
                size = [1, 1, 220, 220, 84]
                align_corners = False
                grid = torch.nn.functional.affine_grid(theta=theta, size=size, align_corners=align_corners)
                return g.op(
                    "onnx::Constant",
                    value_t=torch.tensor(grid.numpy().flatten().tolist())
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

            # torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
            #     temp_module, args=loaded_image_tensor, opset_version=opset_version
            #     )
            # print(torch_script_graph)
            # print(set(unconvertible_ops))

            f = io.BytesIO()
            torch.onnx.export(temp_module, loaded_image_tensor, f, verbose=False, opset_version=18)
            onnx_model = onnx.load_model_from_string(f.getvalue())
            print(onnx_model)
            onnx.save(onnx_model, "c:/temp/test_preprocessing.onnx")

            import onnxruntime as ort
            model_input_names = [i.name for i in onnx_model.graph.input]
            input_dict = dict(zip(model_input_names, [i.cpu().numpy() for i in [loaded_image_tensor]]))
            ort_sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
            onnx_out = ort_sess.run(None, input_dict)

            for r1, r2 in zip(load_and_preprocessed_data["pred"], onnx_out):
                torch.testing.assert_allclose(r1.cpu(), r2, rtol=rtol, atol=atol)  # type: ignore

    def test_postprocessing_to_onnx(self):
        device = torch.device("cpu")
        data_dir = "C:/LiqunWA/MONAI/data/Task09_Spleen"
        images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
        load_and_preprocess_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )
        load_and_preprocess_transforms_ds = Dataset(data=data, transform=load_and_preprocess_transforms)
        load_and_preprocessed_tensor = load_and_preprocess_transforms_ds[0]["image"].as_tensor().to(device)

        post_transforms = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=load_and_preprocess_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                    device="cpu",
                ),
                AsDiscreted(keys="pred", argmax=True, to_onehot=2),
                # AsDiscreted(keys="label", to_onehot=2),
            ]
        )

        opset_version = 18
        with torch.no_grad():
            image_meta_dict = load_and_preprocess_transforms_ds[0]["image_meta_dict"]
            load_and_preprocessed_data = load_and_preprocess_transforms_ds[0]
            load_and_preprocessed_image = load_and_preprocessed_data["image"].to(device)
            load_and_preprocessed_image = torch.unsqueeze(load_and_preprocessed_image, dim=0)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ).to(device)

            load_and_preprocessed_data["pred"] = sliding_window_inference(load_and_preprocessed_image, roi_size, sw_batch_size, model)
            load_and_preprocessed_data["pred"] = torch.squeeze(load_and_preprocessed_data["pred"], dim=0)
            infered_pred_tensor = load_and_preprocessed_data["pred"].as_tensor().to(device)
            load_and_preprocessed_image = torch.squeeze(load_and_preprocessed_image, dim=0)
            post_processed_data = post_transforms(load_and_preprocessed_data)

            temp_module = TempModule(postprocessing=post_transforms, postprocessing_key="pred", image_meta_dict=image_meta_dict)
            torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(temp_module, args=infered_pred_tensor, opset_version=opset_version)
            # print(torch_script_graph)
            # print(set(unconvertible_ops))
            assert(len(unconvertible_ops) == 0)

            f = io.BytesIO()
            torch.onnx.export(temp_module, infered_pred_tensor, f, verbose=False, opset_version=18)
            onnx_model = onnx.load_model_from_string(f.getvalue())
            print(onnx_model)
            onnx.save(onnx_model, "c:/temp/test_postprocessing.onnx")

            import onnxruntime as ort
            model_input_names = [i.name for i in onnx_model.graph.input]
            input_dict = dict(zip(model_input_names, [i.cpu().numpy() for i in [infered_pred_tensor]]))
            ort_sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
            onnx_out = ort_sess.run(None, input_dict)

            torch.testing.assert_allclose(post_processed_data["pred"].as_tensor(), onnx_out[0])
            for r1, r2 in zip(post_processed_data["pred"], onnx_out):
                torch.testing.assert_allclose(r1.cpu(), r2, rtol=rtol, atol=atol)  # type: ignore


    def test_default(self):
        device = torch.device("cpu")
        data_dir = "C:/LiqunWA/MONAI/data/Task09_Spleen"
        images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))[0:1]
        labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))[0:1]
        data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]

        test_org_transforms = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

        test_org_ds = Dataset(data=data, transform=test_org_transforms)

        test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

        post_transforms = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=test_org_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True, to_onehot=2),
                SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="C:/Temp/", output_postfix="seg", resample=False),
            ]
        )

        device = torch.device("cpu")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)
        model.load_state_dict(torch.load("C:/LiqunWA/MONAI/data/best_metric_model.pth"))
        model.eval()

        with torch.no_grad():
            for test_data in test_org_loader:
                test_inputs = test_data["image"].to(device)
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

                use_decollate = False
                if use_decollate:
                    test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                else:
                    for i in decollate_batch(test_data):
                        postprocessed_pred = post_transforms(i)
                        print(postprocessed_pred["pred"].shape)

    @parameterized.expand([TEST_CASE_1])
    def test_evaluator_to_onnx(self, input_params, decollate, expected):
        data = [
            {"image": torch.tensor([[[[2.0], [3.0]]]]), "filename": ["test1"]},
            {"image": torch.tensor([[[[6.0], [8.0]]]]), "filename": ["test2"]},
        ]
        # set up engine, PostProcessing handler works together with postprocessing transforms of engine
        inferer = SlidingWindowInferer([2, 2])
        engine = SupervisedEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=data,
            epoch_length=2,
            network=torch.nn.PReLU(),
            inferer=inferer,
            postprocessing=Compose([Activationsd(keys="pred", sigmoid=True)]),
            val_handlers=[PostProcessing(**input_params)],
            decollate=decollate,
        )

        class TempModule(torch.nn.Module):
            def __init__(self,
                         network: torch.nn.Module,
                         preprocessing: Callable | None = None,
                         postprocessing: Transform | None = None,
                         ):
                super().__init__()
                self.network = network
                self.preprocessing = preprocessing
                self.postprocessing = postprocessing
        
            def forward(self, x):
                if self.preprocessing:
                    x = self.preprocessing(x)
                y = self.network(x)
                if self.postprocessing:
                    y = self.postprocessing({"pred": y})
                return y

        temp_module = TempModule(engine.network, postprocessing=Compose([Activationsd(keys="pred", sigmoid=True)]))
        temp_module.eval()
        for param in temp_module.network.parameters():
            param.requires_grad = False
        f = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(temp_module, data[0]["image"], f, verbose=False, opset_version=11)
        onnx_model = onnx.load_model_from_string(f.getvalue())
        print(onnx_model)
        onnx.save(onnx_model, "c:/temp/eval.onnx")

    def test_spleen_segmentor_to_onnx(self):

        device = torch.device("cpu")

        data_dir = "C:/LiqunWA/MONAI/data/Task09_Spleen"
        test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
        test_data = [{"image": image} for image in test_images]

        test_org_transforms = Compose(
            [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

        test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

        test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

        post_transforms = Compose(
            [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
            ]
        )

        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(device)

        input_data = None

        with torch.no_grad():
            for test_data in test_org_loader:
                test_inputs = test_data["image"].to(device)
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                test_input_tensor = test_inputs.as_tensor().to(device)
                print(type(test_input_tensor))
                print(test_input_tensor.shape)
                test_data["pred"] = sliding_window_inference(test_input_tensor, roi_size, sw_batch_size, model)

                # test_data = [post_transforms(i) for i in decollate_batch(test_data)]

                temp_module = TempModule(model, preprocessing = None, postprocessing=post_transforms)
                # mode_to_export = torch.jit.script(temp_module)
                f = io.BytesIO()
                with torch.no_grad():
                    torch.onnx.export(temp_module, test_input_tensor, f, verbose=False, opset_version=11)
                onnx_model = onnx.load_model_from_string(f.getvalue())
                print(onnx_model)
                onnx.save(onnx_model, "c:/temp/eval2.onnx")

                import onnxruntime as ort
                model_input_names = [i.name for i in onnx_model.graph.input]
                input_dict = dict(zip(model_input_names, [i.cpu().numpy() for i in [test_input_tensor]]))
                ort_sess = ort.InferenceSession(
                    onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                onnx_out = ort_sess.run(None, input_dict)

                for r1, r2 in zip(test_data["pred"], onnx_out):
                    torch.testing.assert_allclose(r1.cpu(), r2, rtol=rtol, atol=atol)  # type: ignore


                break

            for test_data in test_org_loader:
                test_inputs = test_data["image"].to(device)
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                break


if __name__ == "__main__":
    unittest.main()
