/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
//#include <Python.h>
//#include <gflags/gflags.h>
//#include <glog/logging.h>
#include "rasterize_points.h"

namespace py = pybind11;
void init_glog(int* argc, char** argv) {
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //init_glog(&Py_GetArgc(), &Py_GetArgv());
  init_glog(nullptr, nullptr);
  /*
  LOG(INFO) << "This is an info message";
  LOG(WARNING) << "This is a warning message";
  LOG(ERROR) << "This is an error message";
  VLOG(1) << "This message is logged when verbose level is set to 1 or lower";
  */
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  //google::ShutdownGoogleLogging();
}