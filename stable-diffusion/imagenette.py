# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Imagenette dataset."""

import os
import json

import datasets


_HOMEPAGE = "https://github.com/fastai/imagenette"

_LICENSE = "Apache License 2.0"

_CITATION = """\
@software{Howard_Imagenette_2019,
    title={Imagenette: A smaller subset of 10 easily classified classes from Imagenet},
    author={Jeremy Howard},
    year={2019},
    month={March},
    publisher = {GitHub},
    url = {https://github.com/fastai/imagenette}
}
"""

_DESCRIPTION = """\
Imagenette is a subset of 10 easily classified classes from Imagenet
(tench, English springer, cassette player, chain saw, church, French
horn, garbage truck, gas pump, golf ball, parachute).
"""

_LABEL_MAP = [
    'n01440764',
    'n02102040',
    'n02979186',
    'n03000684',
    'n03028079',
    'n03394916',
    'n03417042',
    'n03425413',
    'n03445777',
    'n03888257',
]

_REPO = "https://huggingface.co/datasets/frgfm/imagenette/resolve/main/metadata"


class ImagenetteConfig(datasets.BuilderConfig):
    """BuilderConfig for Imagette."""

    def __init__(self, data_url, metadata_urls, **kwargs):
        """BuilderConfig for Imagette.
        Args:
          data_url: `string`, url to download the zip file from.
          matadata_urls: dictionary with keys 'train' and 'validation' containing the archive metadata URLs
          **kwargs: keyword arguments forwarded to super.
        """
        super(ImagenetteConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_url = data_url
        self.metadata_urls = metadata_urls


class Imagenette(datasets.GeneratorBasedBuilder):
    """Imagenette dataset."""

    BUILDER_CONFIGS = [
        ImagenetteConfig(
            name="full_size",
            description="All images are in their original size.",
            data_url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            metadata_urls={
                "train": f"{_REPO}/imagenette2/train.txt",
                "validation": f"{_REPO}/imagenette2/val.txt",
            },
        ),
        ImagenetteConfig(
            name="320px",
            description="All images were resized on their shortest side to 320 pixels.",
            data_url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            metadata_urls={
                "train": f"{_REPO}/imagenette2-320/train.txt",
                "validation": f"{_REPO}/imagenette2-320/val.txt",
            },
        ),
        ImagenetteConfig(
            name="160px",
            description="All images were resized on their shortest side to 160 pixels.",
            data_url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            metadata_urls={
                "train": f"{_REPO}/imagenette2-160/train.txt",
                "validation": f"{_REPO}/imagenette2-160/val.txt",
            },
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION + self.config.description,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=[
                            "tench",
                            "English springer",
                            "cassette player",
                            "chain saw",
                            "church",
                            "French horn",
                            "garbage truck",
                            "gas pump",
                            "golf ball",
                            "parachute",
                        ]
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(self.config.data_url)
        metadata_paths = dl_manager.download(self.config.metadata_urls)
        archive_iter = dl_manager.iter_archive(archive_path)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": archive_iter,
                    "metadata_path": metadata_paths["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images": archive_iter,
                    "metadata_path": metadata_paths["validation"],
                },
            ),
        ]

    def _generate_examples(self, images, metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            files_to_keep = set(f.read().split("\n"))
        idx = 0
        for file_path, file_obj in images:
            if file_path in files_to_keep:
                label = _LABEL_MAP.index(file_path.split("/")[-2])
                yield idx, {
                    "image": {"path": file_path, "bytes": file_obj.read()},
                    "label": label,
                }
                idx += 1
