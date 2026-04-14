# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages

setup(
    name="madeline",
    version="0.1.0",
    description="Forward-Pass Parameter Caching for DeepSpeed ZeRO-3",
    author="Ruihan Zhang",
    packages=find_packages(exclude=["tests*", "experiments*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "experiment": [
            "transformers>=4.30.0",
            "datasets",
        ],
    },
)
