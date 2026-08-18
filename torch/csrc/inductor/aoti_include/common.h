#pragma once

#include <array>
#include <filesystem>
#include <optional>

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model.h>

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/align.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
