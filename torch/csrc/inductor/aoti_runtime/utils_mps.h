#pragma once

#ifdef USE_MPS
// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>

namespace torch::aot_inductor {

// MPS exposes a single device and a single default stream, so neither guard
// has anything to switch. They exist to satisfy the codegen that emits a
// guard around every device context.
class AOTIMpsGuard {
 public:
  AOTIMpsGuard(int32_t device_index) {
    check_index(device_index);
  }

  void set_index(int32_t device_index) {
    check_index(device_index);
  }

 private:
  static void check_index(int32_t device_index) {
    if (device_index != 0) {
      throw std::runtime_error("MPS only supports device index 0");
    }
  }
};

class AOTIMpsStreamGuard {
 public:
  AOTIMpsStreamGuard(void* stream, int32_t device_index) {
    (void)stream;
    if (device_index != 0) {
      throw std::runtime_error("MPS only supports device index 0");
    }
  }
};

} // namespace torch::aot_inductor
#endif // USE_MPS
