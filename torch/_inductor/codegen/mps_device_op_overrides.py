from __future__ import annotations

from .common import DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        # MPS kernels take the stream as an opaque token they never
        # dereference, so the stream id stands in for a queue pointer.
        return f"{name} = lambda device_idx: torch.mps.current_stream().stream_id"

    def current_stream(self) -> str:
        return "torch.mps.current_stream()"

    def device_guard(self, device_idx: int) -> str:
        if device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: int) -> str:
        if device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "pass  # MPS set device"

    def synchronize(self) -> str:
        return "torch.mps.synchronize()"

    def kernel_driver(self) -> str:
        # The library must outlive the function it vends, so both are parked in
        # a function-local cache keyed by metallib path.
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
            #include <memory>
            #include <optional>
            #include <string>
            #include <unordered_map>
            #include <vector>

            // The shared emitter casts the out-param to void**, matching CUDA's
            // getter; adapt the accelerator-generic one to that shape.
            static inline AOTITorchError aoti_torch_get_current_mps_stream(
                    int32_t device_index,
                    void** ret_stream) {
                return aoti_torch_get_current_stream(
                    device_index, reinterpret_cast<StreamHandle*>(ret_stream));
            }

            // AOTI appends cubin_dir_; the metallib path is already absolute,
            // so it is accepted and ignored.
            static inline at::native::mps::MetalKernelFunction* loadKernel(
                    const std::string& metallibPath,
                    const std::string& funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string>& cubinDir = std::nullopt) {
                (void)sharedMemBytes;
                (void)cubinDir;
                static std::unordered_map<
                    std::string,
                    std::unique_ptr<at::native::mps::PrecompiledMetalShaderLibrary>> libs;
                static std::unordered_map<
                    std::string,
                    std::shared_ptr<at::native::mps::MetalKernelFunction>> funcs;
                auto key = metallibPath + "#" + funcName;
                auto cached = funcs.find(key);
                if (cached != funcs.end()) {
                    return cached->second.get();
                }
                auto lib = libs.find(metallibPath);
                if (lib == libs.end()) {
                    lib = libs.emplace(
                        metallibPath,
                        std::make_unique<at::native::mps::PrecompiledMetalShaderLibrary>(
                            metallibPath)).first;
                }
                auto fn = lib->second->getKernelFunction(funcName);
                return funcs.emplace(key, std::move(fn)).first->second.get();
            }

            static inline void launchKernel(
                    at::native::mps::MetalKernelFunction* func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    void* stream,
                    const bool is_ptr[] = nullptr,
                    const unsigned scalar_size[] = nullptr,
                    unsigned nargs = 0) {
                (void)sharedMemBytes;
                (void)stream;
                const uint64_t threadsPerGroup =
                    static_cast<uint64_t>(numWarps) * func->getThreadExecutionWidth();
                const std::vector<uint64_t> grid = {
                    static_cast<uint64_t>(gridX) * threadsPerGroup,
                    static_cast<uint64_t>(gridY),
                    static_cast<uint64_t>(gridZ)};
                const std::vector<uint64_t> group = {threadsPerGroup, 1, 1};

                // The emitted MSL signature is [ptr0, ptr1, ..., packed_scalars]:
                // pointers keep their relative order and every scalar is packed,
                // at natural alignment, into one trailing setBytes buffer. See
                // EmitMSLFunc.cpp's argbuf packing, which this must mirror.
                std::vector<unsigned char> packed;
                for (unsigned i = 0; i < nargs; ++i) {
                    if (is_ptr[i]) {
                        continue;
                    }
                    const unsigned sz = scalar_size[i];
                    packed.resize((packed.size() + sz - 1) / sz * sz);
                    const auto* bytes = static_cast<const unsigned char*>(args[i]);
                    packed.insert(packed.end(), bytes, bytes + sz);
                }

                func->runCommandBlock([&] {
                    func->startEncoding();
                    auto handle =
                        reinterpret_cast<AOTIMetalKernelFunctionHandle>(func);
                    unsigned slot = 0;
                    for (unsigned i = 0; i < nargs; ++i) {
                        if (!is_ptr[i]) {
                            continue;
                        }
                        // Tensor slots hold the ADDRESS of an AtenTensorHandle,
                        // per the shared void*[] convention.
                        AOTI_TORCH_ERROR_CODE_CHECK(
                            aoti_torch_mps_set_arg_tensor(
                                handle, slot++,
                                *static_cast<AtenTensorHandle*>(args[i])));
                    }
                    if (!packed.empty()) {
                        func->setArg(slot, packed.data(), packed.size());
                    }
                    func->dispatch(grid, group);
                });
            }
        """

    def cpp_kernel_type(self) -> str:
        return "at::native::mps::MetalKernelFunction*"

    def launch_needs_arg_kinds(self) -> bool:
        return True

    def cpp_device_guard(self) -> str:
        return "torch::aot_inductor::AOTIMpsGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "torch::aot_inductor::AOTIMpsGuard"

    def cpp_stream_guard(self) -> str:
        return "torch::aot_inductor::AOTIMpsStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "torch::aot_inductor::AOTIMpsStreamGuard"

    def cpp_stream_type(self) -> str:
        # Must match DeviceStreamType in aoti_runtime/device_utils.h, which is
        # void* for every device without its own branch. MPS kernels never
        # dereference the stream.
        return "void*"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_mps_stream"

    def cpp_device_ptr(self) -> str:
        # Metal binds allocations, not addresses, so the launch path carries
        # tensor handles rather than data pointers.
        return "AtenTensorHandle"

    def kernel_header(self) -> str:
        return """
        #include <torch/csrc/inductor/aoti_runtime/utils_mps.h>
        """


register_device_op_overrides("mps", MPSDeviceOpOverrides())
