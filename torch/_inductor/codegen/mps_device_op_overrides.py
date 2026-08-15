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
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
