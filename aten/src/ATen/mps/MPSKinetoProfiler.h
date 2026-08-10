//  Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <string>

namespace at::mps {

// Records one completed Metal command buffer for the Kineto trace.
//
// MPS has no CUPTI-style device tracer, so torch::profiler asked Kineto for
// MPS activities and got CPU ones back (kineto_shim.cpp), which is why device
// times came back zero. Metal reports GPUStartTime/GPUEndTime on the command
// buffer completion handler; those are absolute and already on
// CLOCK_MONOTONIC_RAW, the clock Kineto uses, so they can be forwarded as-is.
//
// isMPSKinetoProfilingActive() is checked before doing any of the string work
// in the completion handler, which runs on every dispatch.
bool isMPSKinetoProfilingActive();

void recordMPSKinetoActivity(
    const std::string& name,
    uint64_t correlationId,
    double gpuStartSeconds,
    double gpuEndSeconds,
    bool isCopy);

// Installs the MPS profiler factory with libkineto. Idempotent; safe to call
// before libkineto is initialized.
void registerMPSKinetoProfiler();

} // namespace at::mps
