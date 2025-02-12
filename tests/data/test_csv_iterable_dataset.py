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
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from monai.data import CSVIterableDataset, DataLoader
from monai.transforms import ToNumpyd
from tests.test_utils import skip_if_windows


@skip_if_windows
class TestCSVIterableDataset(unittest.TestCase):
    def test_values(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_data1 = [
                ["subject_id", "label", "image", "ehr_0", "ehr_1", "ehr_2"],
                ["s000000", 5, "./imgs/s000000.png", 2.007843256, 2.29019618, 2.054902077],
                ["s000001", 0, "./imgs/s000001.png", 6.839215755, 6.474509716, 5.862744808],
                ["s000002", 4, "./imgs/s000002.png", 3.772548914, 4.211764812, 4.635294437],
                ["s000003", 1, "./imgs/s000003.png", 3.333333254, 3.235294342, 3.400000095],
                ["s000004", 9, "./imgs/s000004.png", 6.427451134, 6.254901886, 5.976470947],
            ]
            test_data2 = [
                ["subject_id", "ehr_3", "ehr_4", "ehr_5", "ehr_6", "ehr_7", "ehr_8"],
                ["s000000", 3.019608021, 3.807843208, 3.584313869, 3.141176462, 3.1960783, 4.211764812],
                ["s000001", 5.192157269, 5.274509907, 5.250980377, 4.647058964, 4.886274338, 4.392156601],
                ["s000002", 5.298039436, 9.545097351, 12.57254887, 6.799999714, 2.1960783, 1.882352948],
                ["s000003", 3.164705753, 3.086274624, 3.725490093, 3.698039293, 3.698039055, 3.701960802],
                ["s000004", 6.26274538, 7.717647076, 9.584313393, 6.082352638, 2.662744999, 2.34117651],
            ]
            test_data3 = [
                ["subject_id", "ehr_9", "ehr_10", "meta_0", "meta_1", "meta_2"],
                ["s000000", 6.301961422, 6.470588684, "TRUE", "TRUE", "TRUE"],
                ["s000001", 5.219608307, 7.827450752, "FALSE", "TRUE", "FALSE"],
                ["s000002", 1.882352948, 2.031372547, "TRUE", "FALSE", "TRUE"],
                ["s000003", 3.309803963, 3.729412079, "FALSE", "FALSE", "TRUE"],
                ["s000004", 2.062745094, 2.34117651, "FALSE", "TRUE", "TRUE"],
            ]

            def prepare_csv_file(data, filepath):
                with open(filepath, "a") as f:
                    for d in data:
                        f.write((",".join([str(i) for i in d])) + "\n")

            filepath1 = os.path.join(tempdir, "test_data1.csv")
            filepath2 = os.path.join(tempdir, "test_data2.csv")
            filepath3 = os.path.join(tempdir, "test_data3.csv")
            filepaths = [filepath1, filepath2, filepath3]
            prepare_csv_file(test_data1, filepath1)
            prepare_csv_file(test_data2, filepath2)
            prepare_csv_file(test_data3, filepath3)

            # test single CSV file
            dataset = CSVIterableDataset(filepath1, shuffle=False)
            count = 0
            for item in dataset:
                count += 1
                if count == 3:
                    self.assertDictEqual(
                        {k: round(v, 4) if not isinstance(v, str) else v for k, v in item.items()},
                        {
                            "subject_id": "s000002",
                            "label": 4,
                            "image": "./imgs/s000002.png",
                            "ehr_0": 3.7725,
                            "ehr_1": 4.2118,
                            "ehr_2": 4.6353,
                        },
                    )
                    break
            self.assertEqual(count, 3)
            dataset.close()

            # test reset iterables
            dataset.reset(src=filepath3)
            count = 0
            for i, item in enumerate(dataset):
                count += 1
                if i == 4:
                    self.assertEqual(item["meta_0"], False)
            self.assertEqual(count, 5)
            dataset.close()

            # test multiple CSV files, join tables with kwargs
            dataset = CSVIterableDataset(filepaths, on="subject_id", shuffle=False)
            count = 0
            for item in dataset:
                count += 1
                if count == 4:
                    self.assertDictEqual(
                        {k: round(v, 4) if not isinstance(v, (str, np.bool_)) else v for k, v in item.items()},
                        {
                            "subject_id": "s000003",
                            "label": 1,
                            "image": "./imgs/s000003.png",
                            "ehr_0": 3.3333,
                            "ehr_1": 3.2353,
                            "ehr_2": 3.4000,
                            "ehr_3": 3.1647,
                            "ehr_4": 3.0863,
                            "ehr_5": 3.7255,
                            "ehr_6": 3.6980,
                            "ehr_7": 3.6980,
                            "ehr_8": 3.7020,
                            "ehr_9": 3.3098,
                            "ehr_10": 3.7294,
                            "meta_0": False,
                            "meta_1": False,
                            "meta_2": True,
                        },
                    )
            self.assertEqual(count, 5)
            dataset.close()

            # test selected columns and chunk size
            dataset = CSVIterableDataset(
                src=filepaths, chunksize=2, col_names=["subject_id", "image", "ehr_1", "ehr_7", "meta_1"], shuffle=False
            )
            count = 0
            for item in dataset:
                count += 1
                if count == 4:
                    self.assertDictEqual(
                        {k: round(v, 4) if not isinstance(v, (str, np.bool_)) else v for k, v in item.items()},
                        {
                            "subject_id": "s000003",
                            "image": "./imgs/s000003.png",
                            "ehr_1": 3.2353,
                            "ehr_7": 3.6980,
                            "meta_1": False,
                        },
                    )
            self.assertEqual(count, 5)
            dataset.close()

            # test group columns
            dataset = CSVIterableDataset(
                src=filepaths,
                col_names=["subject_id", "image", *[f"ehr_{i}" for i in range(11)], "meta_0", "meta_1", "meta_2"],
                col_groups={"ehr": [f"ehr_{i}" for i in range(11)], "meta12": ["meta_1", "meta_2"]},
                shuffle=False,
            )
            count = 0
            for item in dataset:
                count += 1
                if count == 4:
                    np.testing.assert_allclose(
                        [round(i, 4) for i in item["ehr"]],
                        [3.3333, 3.2353, 3.4000, 3.1647, 3.0863, 3.7255, 3.6980, 3.6980, 3.7020, 3.3098, 3.7294],
                    )
                    np.testing.assert_allclose(item["meta12"], [False, True])
            self.assertEqual(count, 5)
            dataset.close()

            # test transform
            dataset = CSVIterableDataset(
                chunksize=2,
                buffer_size=4,
                src=filepaths,
                col_groups={"ehr": [f"ehr_{i}" for i in range(5)]},
                transform=ToNumpyd(keys="ehr"),
                shuffle=True,
                seed=123,
            )
            expected = [
                [6.8392, 6.4745, 5.8627, 5.1922, 5.2745],
                [3.3333, 3.2353, 3.4000, 3.1647, 3.0863],
                [3.7725, 4.2118, 4.6353, 5.298, 9.5451],
                [6.4275, 6.2549, 5.9765, 6.2627, 7.7176],
                [2.0078, 2.2902, 2.0549, 3.0196, 3.8078],
            ]
            count = 0
            for item, exp in zip(dataset, expected):
                count += 1
                self.assertTrue(isinstance(item["ehr"], np.ndarray))
                np.testing.assert_allclose(np.around(item["ehr"], 4), exp)
            self.assertEqual(count, 5)
            dataset.close()

            # test multiple processes loading
            dataset = CSVIterableDataset(filepath1, transform=ToNumpyd(keys="label"), shuffle=False)
            # set num workers = 0 for mac / win
            num_workers = 2 if sys.platform == "linux" else 0
            dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=2)
            count = 0
            for item in dataloader:
                count += 1
                # test the last item which only has 1 data
                if len(item) == 1:
                    self.assertListEqual(item["subject_id"], ["s000002"])
                    np.testing.assert_allclose(item["label"], [4])
                    self.assertListEqual(item["image"], ["./imgs/s000002.png"])
            self.assertEqual(count, 3)
            dataset.close()

            # test iterable stream
            iters = pd.read_csv(filepath1, chunksize=1000)
            dataset = CSVIterableDataset(src=iters, shuffle=False)
            count = 0
            for item in dataset:
                count += 1
                if count == 3:
                    self.assertDictEqual(
                        {k: round(v, 4) if not isinstance(v, str) else v for k, v in item.items()},
                        {
                            "subject_id": "s000002",
                            "label": 4,
                            "image": "./imgs/s000002.png",
                            "ehr_0": 3.7725,
                            "ehr_1": 4.2118,
                            "ehr_2": 4.6353,
                        },
                    )
                    break
            self.assertEqual(count, 3)
            dataset.close()

            # test multiple iterable streams, join tables with kwargs
            iters = [pd.read_csv(i, chunksize=1000) for i in filepaths]
            dataset = CSVIterableDataset(src=iters, on="subject_id", shuffle=False)
            count = 0
            for item in dataset:
                count += 1
                if count == 4:
                    self.assertEqual(item["subject_id"], "s000003")
                    self.assertEqual(item["label"], 1)
                    self.assertEqual(round(item["ehr_0"], 4), 3.3333)
                    self.assertEqual(item["meta_0"], False)
            self.assertEqual(count, 5)
            # manually close the pre-loaded iterables instead of `dataset.close()`
            for i in iters:
                i.close()


if __name__ == "__main__":
    unittest.main()
