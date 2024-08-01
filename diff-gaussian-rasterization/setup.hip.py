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
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CppExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args=["-I " + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                                "-mprintf-kind=buffered", "-lhip_hcc", '-g', '-O3',],
            #include_dirs=[],
            #extra_link_args=[ '-lgflags', '-lglog']
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
