//  Copyright © 2022 Apple Inc.

#pragma once

#include <cstdint>
#include <utility>

#include <ATen/mps/MPSDevice.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
// Apple framework headers emit deprecation warnings from CarbonCore and
// missing-attribute warnings from MPSGraph on recent macOS SDKs.
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wobjc-property-no-attribute")
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLBuffer> MTLBuffer_t;
#else
#include <dispatch/dispatch.h>
typedef void* MPSCommandBuffer_t;
typedef void* MPSGraph;
typedef void* MPSGraphExecutionDescriptor;
typedef void* MPSGraphCompilationDescriptor;
typedef void* MTLCommandQueue_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLSharedEvent_t;
typedef void* MTLDevice_t;
typedef void* MTLBuffer_t;
typedef void* MTLCommandBufferHandler;
typedef void* NSDictionary;
#define nil NULL
#endif

namespace at::mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

enum class SyncType {
  NONE, // no commit to command buffer
  COMMIT, // commit and flush the command buffer
  COMMIT_AND_WAIT, // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
  COMMIT_ADAPTIVE, // commit adaptively based on available memory
};

class TORCH_API MPSStream {
 public:
  enum Unchecked { UNCHECKED };

  /// Construct a MPSStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  ~MPSStream();

  MTLCommandQueue_t commandQueue() const {
    return _commandQueue;
  }

  dispatch_queue_t queue() const {
    return _serialQueue;
  }

  MPSCommandBuffer_t commandBuffer();
  MTLComputeCommandEncoder_t commandEncoder();
  void endKernelCoalescing();
  void synchronize(SyncType syncType);

  /// Returns true when every command buffer this stream has submitted has
  /// finished, i.e. synchronize() would not block.
  bool query() const;
  void copy(MTLBuffer_t srcBuffer,
            MTLBuffer_t dstBuffer,
            size_t length,
            size_t srcOffset,
            size_t dstOffset,
            uint64_t profileId,
            SyncType syncType = SyncType::NONE);
  void copy_and_sync(MTLBuffer_t srcBuffer,
                     MTLBuffer_t dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     bool non_blocking,
                     uint64_t profileId);
  void executeMPSGraph(MPSGraph* mpsGraph,
                       NSDictionary* feeds,
                       NSDictionary* results,
                       SyncType syncType = SyncType::NONE);
  void addCompletedHandler(MTLCommandBufferHandler block);

  /// Get the MPS device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return _stream.device_index();
  }

  MTLCommandQueue_t stream() const {
    return _commandQueue;
  }

  MTLDevice_t device() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return _stream;
  }

  MTLBuffer_t getErrorBuffer();
  void checkLastError();

  // Timing pin: while pinned, involuntary commits (adaptive / watermark driven
  // and commitAndContinue) are suppressed so that the current command buffer is
  // not split between a start and end timing event. This guarantees both events
  // of an inductor benchmark pair land on the same command buffer, so their GPU
  // timestamps cannot invert under concurrent GPU load. Explicit COMMIT and
  // COMMIT_AND_WAIT (the benchmark's own end-of-pair synchronize) still commit.
  void pinTiming() {
    ++_timingPinned;
  }
  void unpinTiming() {
    if (_timingPinned > 0) {
      --_timingPinned;
    }
  }
  bool isTimingPinned() const {
    return _timingPinned > 0;
  }
  // Brackets a timed event pair. The first timing record opens the region and
  // pins; this returns true. The second timing record returns false to signal
  // the caller it must unpin (via unpinTiming()) AFTER the end signal is encoded
  // onto the shared command buffer. Driven from MPSEvent::recordLocked.
  bool openOrCloseTimedPair() {
    _inTimedPair = !_inTimedPair;
    if (_inTimedPair) {
      pinTiming();
      return true;
    }
    return false;
  }
  // Clears any open timed region. Used as an exception-safety backstop on a full
  // GPU drain so an interrupted pair cannot leak commit suppression.
  void resetTimingPin() {
    _timingPinned = 0;
    _inTimedPair = false;
  }

 private:
  Stream _stream;
  MTLCommandQueue_t _commandQueue = nil;
  MPSCommandBuffer_t _commandBuffer = nil;
  MPSCommandBuffer_t _prevCommandBuffer = nil;
  MTLComputeCommandEncoder_t _commandEncoder = nil;
  MPSGraphExecutionDescriptor* _executionDescriptor = nil;
  MPSGraphCompilationDescriptor* _compilationDescriptor = nil;
  dispatch_queue_t _serialQueue = nullptr;
  // CommitAndContinue is enabled by default
  bool _enableCommitAndContinue = true;
  bool _kinetoHandlerAttached = false;
  // >0 while a timed region (start..end event pair) is open; suppresses
  // involuntary command buffer commits so the pair stays on one buffer.
  int _timingPinned = 0;
  // false between pairs, true after a start timing record and before its end.
  bool _inTimedPair = false;
  // Buffer that contains last raised error
  MTLBuffer_t _errorBuffer = nil;

  // use synchronize() to access any of these commit functions outside MPSStream
  void commit();
  void commitAndWait();
  void commitAndContinue();
  void flush();
  void addKinetoCompletedHandler();
};

/**
 * Get the current MPS stream for this thread. Returns the default stream if no
 * other stream has been set with `setCurrentMPSStream()`.
 */
TORCH_API MPSStream* getCurrentMPSStream();

/**
 * Set the current MPS stream for this thread. Kernels that call
 * `getCurrentMPSStream()` will enqueue their work onto this stream. Passing
 * nullptr sets to the default stream.
 */
TORCH_API void setCurrentMPSStream(MPSStream* stream);

/**
 * Get the default MPS stream
 */
TORCH_API MPSStream* getDefaultMPSStream();

/**
 * Get a stream from the pool. There are 32 streams in the pool which live for
 * the lifetime of a process. The stream returned by this function is chosen
 * in round-robin order. Note: The default stream is not in the pool.
 */
TORCH_API MPSStream* getStreamFromPool();

/**
 * Get the MPSStream carrying a given StreamId: 0 is the default stream, 1..32
 * index the pool. Turns a c10::Stream that crossed an API boundary back into
 * the stream it names, which is what MPSGuardImpl::exchangeStream needs.
 */
TORCH_API MPSStream* getStreamFromId(c10::StreamId id);

/**
 * Synchronize the default stream and any pool streams created so far.
 */
TORCH_API void synchronizeAllMPSStreams(SyncType syncType);

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

class TORCH_API MPSStreamImpl {
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};

#ifdef __OBJC__
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());
#endif
} // namespace at::mps
