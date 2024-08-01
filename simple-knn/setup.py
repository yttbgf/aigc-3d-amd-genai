#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))
os.environ["CC"] = "hipcc"
os.environ["CXX"] = "hipcc"

cxx_compiler_flags = ["-mprintf-kind=buffered", "-lhip_hcc", '-g', '-O3']

setup(
    name="simple_knn",
    ext_modules=[
        CppExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu",
            "simple_knn.hip",
            "ext.cpp"],
            extra_compile_args=cxx_compiler_flags
        )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
