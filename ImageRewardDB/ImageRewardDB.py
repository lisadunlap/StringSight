# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Loading scripts for ImageRewardDB."""


import pandas as pd
import json
import os

import datasets
from huggingface_hub import hf_hub_url


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{xu2023imagereward,
      title={ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation}, 
      author={Jiazheng Xu and Xiao Liu and Yuchen Wu and Yuxuan Tong and Qinkai Li and Ming Ding and Jie Tang and Yuxiao Dong},
      year={2023},
      eprint={2304.05977},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

# You can copy an official description
_DESCRIPTION = """\
ImageRewardDB is a comprehensive text-to-image comparison dataset, focusing on text-to-image human preference. \
It consists of 137k pairs of expert comparisons, based on text prompts and corresponding model outputs from DiffusionDB. \
To build the ImageRewadDB, we design a pipeline tailored for it, establishing criteria for quantitative assessment and \
annotator training, optimizing labeling experience, and ensuring quality validation. \
"""

_HOMEPAGE = "https://huggingface.co/datasets/THUDM/ImageRewardDB"
_VERSION = datasets.Version("1.0.0")

_LICENSE = "Apache License 2.0"

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_REPO_ID = "THUDM/ImageRewardDB"
_URLS = {}
_PART_IDS = {
    "train": 32,
    "validation": 2,
    "test": 2
}

for name in list(_PART_IDS.keys()):
    _URLS[name] = {}
    for i in range(1, _PART_IDS[name]+1):
        _URLS[name][i] = hf_hub_url(
            _REPO_ID,
            filename=f"images/{name}/{name}_{i}.zip",
            repo_type="dataset"
        )
    _URLS[name]["metadata"] = hf_hub_url(
        _REPO_ID,
        filename=f"metadata-{name}.parquet",
        repo_type="dataset"
    )

class ImageRewardDBConfig(datasets.BuilderConfig):
    '''BuilderConfig for ImageRewardDB'''
    
    def __init__(self, part_ids, **kwargs):
        '''BuilderConfig for ImageRewardDB
        Args:
            part_ids([int]): A list of part_ids.
            **kwargs: keyword arguments forwarded to super
        '''
        super(ImageRewardDBConfig, self).__init__(version=_VERSION, **kwargs)
        self.part_ids = part_ids

class ImageRewardDB(datasets.GeneratorBasedBuilder):
    """A dataset of 137k expert comparisons to date, demonstrating the text-to-image human preference."""

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    
    BUILDER_CONFIGS = []
    
    for num_k in [1,2,4,8]:
        part_ids = {
            "train": 4*num_k,
            "validation": 2,
            "test": 2
        }
        BUILDER_CONFIGS.append(
            ImageRewardDBConfig(name=f"{num_k}k_group", part_ids=part_ids, description=f"This is a {num_k}k-scale groups of ImageRewardDB")
        )
        BUILDER_CONFIGS.append(
            ImageRewardDBConfig(name=f"{num_k}k", part_ids=part_ids, description=f"This is a {num_k}k-scale ImageRewardDB")
        )
        BUILDER_CONFIGS.append(
            ImageRewardDBConfig(name=f"{num_k}k_pair", part_ids=part_ids, description=f"This is a {num_k}k-scale pairs of ImageRewardDB")
        )

    DEFAULT_CONFIG_NAME = "8k"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if "group" in self.config.name:
            features = datasets.Features(
                {
                    "prompt_id": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "classification": datasets.Value("string"),
                    "image": datasets.Sequence(datasets.Image()),
                    "rank": datasets.Sequence(datasets.Value("int8")),
                    "overall_rating": datasets.Sequence(datasets.Value("int8")),
                    "image_text_alignment_rating": datasets.Sequence(datasets.Value("int8")),
                    "fidelity_rating": datasets.Sequence(datasets.Value("int8"))
                }
            )
        elif "pair" in self.config.name:
            features = datasets.Features(
                {
                    "prompt_id": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "classification": datasets.Value("string"),
                    "img_better": datasets.Image(),
                    "img_worse": datasets.Image()
                }
            )
        else:
            features = datasets.Features(
                {
                    "image": datasets.Image(),
                    "prompt_id": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "classification": datasets.Value("string"),
                    "image_amount_in_total": datasets.Value("int8"),
                    "rank": datasets.Value("int8"),
                    "overall_rating": datasets.Value("int8"),
                    "image_text_alignment_rating": datasets.Value("int8"),
                    "fidelity_rating": datasets.Value("int8")
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_dirs = {name: [] for name in list(_PART_IDS.keys())}
        json_paths = {name: [] for name in list(_PART_IDS.keys())}
        metadata_paths = {name: [] for name in list(_PART_IDS.keys())}
        for key in list(self.config.part_ids.keys()):
            for i in range(1, self.config.part_ids[key]+1):
                data_dir = dl_manager.download_and_extract(_URLS[key][i])
                data_dirs[key].append(data_dir)
                json_paths[key].append(os.path.join(data_dir, f"{key}_{i}.json"))
            metadata_paths[key] = dl_manager.download(_URLS[key]["metadata"])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "train",
                    "data_dirs": data_dirs["train"],
                    "json_paths": json_paths["train"],
                    "metadata_path": metadata_paths["train"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "validation",
                    "data_dirs": data_dirs["validation"],
                    "json_paths": json_paths["validation"],
                    "metadata_path": metadata_paths["validation"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "test",
                    "data_dirs": data_dirs["test"],
                    "json_paths": json_paths["test"],
                    "metadata_path": metadata_paths["test"]
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, split, data_dirs, json_paths, metadata_path):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        
        num_data_dirs = len(data_dirs)
        assert num_data_dirs == len(json_paths)
        
        #Iterate throug all extracted zip folders for images
        # metadata_table = pd.read_parquet(metadata_path)
        for index, json_path in enumerate(json_paths):
            json_data = json.load(open(json_path, "r", encoding="utf-8"))
            if "group" in self.config.name or "pair" in self.config.name:
                group_num = 0
                image_path = []
                rank = []
                overall_rating, image_text_alignment_rating, fidelity_rating = [], [], []
                for sample in json_data:
                    if group_num == 0:
                        image_path.clear()
                        rank.clear()
                        overall_rating.clear()
                        image_text_alignment_rating.clear()
                        fidelity_rating.clear()
                        prompt_id = sample["prompt_id"]
                        prompt = sample["prompt"]
                        classification = sample["classification"]
                        image_amount_in_total = sample["image_amount_in_total"]
                    # image_path.append(sample["image_path"])
                    image_path.append(os.path.join(data_dirs[index], str(sample["image_path"]).split("/")[-1]))
                    rank.append(sample["rank"])
                    overall_rating.append(sample["overall_rating"])
                    image_text_alignment_rating.append(sample["image_text_alignment_rating"])
                    fidelity_rating.append(sample["fidelity_rating"])
                    group_num += 1
                    if group_num == image_amount_in_total:
                        group_num = 0
                        if "group" in self.config.name:
                            yield prompt_id, ({
                                "prompt_id": prompt_id,
                                "prompt": prompt,
                                "classification": classification,
                                "image": [{
                                    "path": image_path[idx],
                                    "bytes": open(image_path[idx], "rb").read()
                                } for idx in range(image_amount_in_total)],
                                "rank": rank,
                                "overall_rating": overall_rating,
                                "image_text_alignment_rating": image_text_alignment_rating,
                                "fidelity_rating": fidelity_rating,
                            })
                        else:
                            for idx in range(image_amount_in_total):
                                for idy in range(idx+1, image_amount_in_total):
                                    if rank[idx] < rank[idy]:
                                        yield prompt_id, ({
                                            "prompt_id": prompt_id,
                                            "prompt": prompt,
                                            "classification": classification,
                                            "img_better": {
                                                "path": image_path[idx],
                                                "bytes": open(image_path[idx], "rb").read()
                                            },
                                            "img_worse": {
                                                "path": image_path[idy],
                                                "bytes": open(image_path[idy], "rb").read()
                                            }
                                        })
                                    elif rank[idx] > rank[idy]:
                                        yield prompt_id, ({
                                            "prompt_id": prompt_id,
                                            "prompt": prompt,
                                            "classification": classification,
                                            "img_better": {
                                                "path": image_path[idy],
                                                "bytes": open(image_path[idy], "rb").read()
                                            },
                                            "img_worse": {
                                                "path": image_path[idx],
                                                "bytes": open(image_path[idx], "rb").read()
                                            }
                                        })
            else:
                for example in json_data:
                    image_path = os.path.join(data_dirs[index], str(example["image_path"]).split("/")[-1])
                    yield example["image_path"], {
                        "image": {
                            "path": image_path,
                            "bytes": open(image_path, "rb").read()
                        },
                        "prompt_id": example["prompt_id"],
                        "prompt": example["prompt"],
                        "classification": example["classification"],
                        "image_amount_in_total": example["image_amount_in_total"],
                        "rank": example["rank"],
                        "overall_rating": example["overall_rating"],
                        "image_text_alignment_rating": example["image_text_alignment_rating"],
                        "fidelity_rating": example["fidelity_rating"]
                    }