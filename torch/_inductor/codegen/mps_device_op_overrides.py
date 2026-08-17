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

            static inline at::native::mps::MetalKernelFunction* loadKernel(
                    const std::string& metallibPath,
                    const std::string& funcName,
                    uint32_t sharedMemBytes) {
                (void)sharedMemBytes;
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
                    StreamHandle stream,
                    const bool is_ptr[],
                    unsigned nargs) {
                (void)sharedMemBytes;
                (void)stream;
                const uint64_t threadsPerGroup =
                    static_cast<uint64_t>(numWarps) * func->getThreadExecutionWidth();
                const std::vector<uint64_t> grid = {
                    static_cast<uint64_t>(gridX) * threadsPerGroup,
                    static_cast<uint64_t>(gridY),
                    static_cast<uint64_t>(gridZ)};
                const std::vector<uint64_t> group = {threadsPerGroup, 1, 1};
                func->runCommandBlock([&] {
                    func->startEncoding();
                    auto handle =
                        reinterpret_cast<AOTIMetalKernelFunctionHandle>(func);
                    for (unsigned i = 0; i < nargs; ++i) {
                        // Every slot holds the ADDRESS of the value, per the
                        // shared void*[] convention.
                        if (is_ptr[i]) {
                            AOTI_TORCH_ERROR_CODE_CHECK(
                                aoti_torch_mps_set_arg_buffer(
                                    handle, i, *static_cast<void**>(args[i])));
                        } else {
                            func->setArg(i, args[i], sizeof(int32_t));
                        }
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
        # aoti_torch_get_current_stream fills a StreamHandle; MPS kernels never
        # dereference it, but the declared type has to match to compile.
        return "StreamHandle"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_mps_stream"

    def cpp_device_ptr(self) -> str:
        return "void*"

    def kernel_header(self) -> str:
        return """
        #include <torch/csrc/inductor/aoti_runtime/utils_mps.h>
        """


register_device_op_overrides("mps", MPSDeviceOpOverrides())
