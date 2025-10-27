# -*- coding: utf-8 -*-

from pathlib import Path

from BayesMonSTR import __VERSION__
from setuptools import find_packages, setup

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="BayesMonSTR",
    version=__VERSION__,
    author="Wenxuan Fan",
    author_email="fanwenxuan@westlake.edu.cn",
    maintainer="WF",
    description="Mosaic STR mutaions calling with bulk and scWGA data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    platforms=["any"],
    keywords=[
        "scDNA-seq",
        "mosaic-mutation",
        "short-tandem-repeat",
        "genomics",
        "life-science",
    ],
    packages=find_packages(),
    package_dir={"BayesMonSTR": "BayesMonSTR"},
    install_requires=[
        i.strip() for i in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "BayesMonSTR=BayesMonSTR.scTRcaller:main",
        ]
    },
)
