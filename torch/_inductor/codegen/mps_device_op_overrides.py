from __future__ import annotations

from .common import DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"{name} = lambda device_idx: None  # MPS has no raw stream"

    def device_guard(self, device_idx: int) -> str:
        assert device_idx == 0
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: int) -> str:
        assert device_idx == 0
        return "pass  # MPS single device"

    def synchronize(self) -> str:
        return "torch.mps.synchronize()"

    def kernel_driver(self) -> str:
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"



register_device_op_overrides("mps", MPSDeviceOpOverrides())
