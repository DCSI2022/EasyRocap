#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) 2024 -2024 WangHaoyu.All Rights Reserved
import setuptools
from setuptools import setup

def get_package_dir():
    pkg_dir = {
        "yolox.tools": "tools",
        "yolox.exp.default": "exps/default",
    }
    return pkg_dir

def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description

def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs

setup(
    name="easyrocap",
    version="1.0.0",
    description="An Easy-to-use Motion Capture System for Robotics",
    author="DCSI2022",
    author_email="spacewang@whu.edu.cn",
    url="https://github.com/DCSI2022/EasyRocap",
    package_dir=get_package_dir(),
    packages=setuptools.find_namespace_packages(),
    python_requires=">=3.6",
    long_description=get_long_description(),
    install_requires=get_install_requirements()
)
