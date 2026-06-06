//  Copyright © 2023 Apple Inc.

#include <ATen/mps/MPSEvent.h>

namespace at::mps {

MPSEvent::MPSEvent(id_t ID, MPSStream* stream, bool enable_timing)
    : m_id(ID), m_enable_timing(enable_timing), m_stream(stream), m_event([stream->device() newSharedEvent]) {}

MPSEvent::~MPSEvent() {
  if (m_event) {
    [m_event release];
    m_event = nil;
  }
  if (m_listener) {
    [m_listener release];
    m_listener = nil;
  }
}

void MPSEvent::recordLocked(bool syncEvent) {
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  // Timing pin: a start timing record opens the pinned region (so involuntary
  // commits triggered by the kernel's allocations between start and end cannot
  // split the command buffer); the matching end timing record closes it after
  // the end signal is encoded below. This guarantees the start and end events
  // share one command buffer, so their GPU timestamps cannot invert.
  bool openedTimedPair = false;
  if (m_enable_timing) {
    openedTimedPair = m_stream->openOrCloseTimedPair();
  }
  ++m_signalCounter;
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  if (m_enable_timing) {
    // Use the command buffer's GPUEndTime: a real GPU timestamp (seconds) for
    // when this buffer finished executing, instead of a CPU monotonic-clock
    // sample taken when the CPU thread observes the shared-event signal. The CPU
    // sample jitters and, for back-to-back fast kernels, ties or inverts across
    // event pairs, which collapses elapsed_time to 0 and makes the inductor
    // autotuner unable to rank choices. GPUEndTime is precise and monotonic.
    // The completed handler fires once the buffer's GPU work (and GPUEndTime) is
    // final; set the completion time and release any waitForCpuSync there, so a
    // reader never observes a stale time. Fall back to the CPU clock only if the
    // GPU timestamp is unavailable (== 0).
    notifyLocked(^(id<MTLSharedEvent>, uint64_t) {});
    // Capture the shared CPU-sync state by value (a strong shared_ptr ref) so
    // this handler, which fires asynchronously on a GPU dispatch thread, can run
    // safely even after this MPSEvent has been destroyed or recycled by the
    // pool. Writing through `this` here would be a use-after-free.
    auto cpuSync = m_cpu_sync;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      double gpuStart = [cb GPUStartTime];
      double gpuEnd = [cb GPUEndTime];
      uint64_t cpuNow = getTime();
      std::lock_guard<std::mutex> lock(cpuSync->mutex);
      cpuSync->start_time =
          gpuStart > 0.0 ? static_cast<uint64_t>(gpuStart * 1e9) : cpuNow;
      cpuSync->completion_time =
          gpuEnd > 0.0 ? static_cast<uint64_t>(gpuEnd * 1e9) : cpuNow;
      cpuSync->completed = true;
      cpuSync->cv.notify_one();
    }];
  }
  [commandBuffer encodeSignalEvent:m_event value:m_signalCounter];
  // Close the pinned region AFTER the end signal is encoded onto the shared
  // buffer, so the buffer is never committed before the end signal lands on it.
  // openedTimedPair is true on the start record (region just opened, keep it
  // pinned) and false on the end record (unpin now). This is reliable:
  // recordLocked always runs to completion for both events of a pair, and the
  // pin counter clamps at zero so a stray unpin can never leak suppression.
  if (m_enable_timing && !openedTimedPair) {
    m_stream->unpinTiming();
  }
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
}

bool MPSEvent::waitLocked(bool syncEvent) {
  // check if event is not recorded yet
  if (m_event.signaledValue >= m_signalCounter) {
    return false;
  }
  // active encoders must end before encoding or waiting
  m_stream->endKernelCoalescing();
  id<MTLCommandBuffer> commandBuffer = m_stream->commandBuffer();
  [commandBuffer encodeWaitForEvent:m_event value:m_signalCounter];
  if (syncEvent) {
    m_stream->synchronize(SyncType::COMMIT);
  }
  return true;
}

bool MPSEvent::notifyLocked(MTLSharedEventNotificationBlock block) {
  // check if event is not recorded yet
  if (m_event.signaledValue >= m_signalCounter) {
    return false;
  }
  if (!m_listener) {
    m_listener = [[MTLSharedEventListener alloc] init];
  }
  [m_event notifyListener:m_listener atValue:m_signalCounter block:block];
  return true;
}

void MPSEvent::record(bool needsLock, bool syncEvent) {
  if (!needsLock) {
    recordLocked(syncEvent);
    return;
  }
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      recordLocked(syncEvent);
    }
  });
}

bool MPSEvent::wait(bool needsLock, bool syncEvent) {
  __block bool waited = false;
  if (!needsLock) {
    return waitLocked(syncEvent);
  }
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      waited = waitLocked(syncEvent);
    }
  });
  return waited;
}

bool MPSEvent::notify(bool needsLock, MTLSharedEventNotificationBlock block) {
  if (!needsLock) {
    return notifyLocked(block);
  }
  __block bool scheduledNotify = false;
  dispatch_sync(m_stream->queue(), ^() {
    @autoreleasepool {
      scheduledNotify = notifyLocked(block);
    }
  });
  return scheduledNotify;
}

void MPSEvent::notifyCpuSync() {
  std::lock_guard<std::mutex> lock(m_cpu_sync->mutex);
  m_cpu_sync->completed = true;
  m_cpu_sync->cv.notify_one();
}

void MPSEvent::waitForCpuSync() {
  auto cpuSync = m_cpu_sync;
  std::unique_lock<std::mutex> lock(cpuSync->mutex);
  cpuSync->cv.wait(lock, [&] { return cpuSync->completed; });
  cpuSync->completed = false;
}

bool MPSEvent::synchronize() {
  // Capture the shared CPU-sync state so the shared-event notify block (which
  // runs on a listener thread) can write the completion time even if this
  // MPSEvent is destroyed before the block fires.
  auto cpuSync = m_cpu_sync;
  bool scheduledNotify = notifyLocked(^(id<MTLSharedEvent>, uint64_t) {
    std::lock_guard<std::mutex> lock(cpuSync->mutex);
    cpuSync->completion_time = getTime();
    cpuSync->completed = true;
    cpuSync->cv.notify_one();
  });

  if (scheduledNotify) {
    waitForCpuSync();
    return true;
  }
  return false;
}

bool MPSEvent::query() const {
  // return false if not recorded or signaled yet
  return m_signalCounter && (m_event.signaledValue >= m_signalCounter);
}

void MPSEvent::reset(MPSStream* stream, bool enable_timing) {
  if (stream != m_stream) {
    m_signalCounter = 0;
    m_event.signaledValue = 0;
    m_stream = stream;
  }
  // Allocate fresh CPU-sync state. Any completion handler still outstanding from
  // a previous recording keeps a strong ref to the OLD state object and writes
  // there harmlessly, so the recycled event cannot be corrupted by a late
  // handler. Resetting also zeroes the timestamps for the new recording.
  m_cpu_sync = std::make_shared<MPSEventCpuSync>();
  m_enable_timing = enable_timing;
};

//-----------------------------------------------------------------
//  MPSEventPool
//-----------------------------------------------------------------

MPSEventPool::MPSEventPool(MPSStream* default_stream) : m_default_stream(default_stream) {
  // default deleter to return the event back to pool after it gets released
  m_default_deleter = [&](MPSEvent* event) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_pool.push(std::unique_ptr<MPSEvent>(event));
  };
}

MPSEventPool::~MPSEventPool() {
  emptyCache();
}

MPSEventPtr MPSEventPool::acquireEvent(bool enable_timing, MPSStream* stream) {
  if (!stream) {
    stream = m_default_stream;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_pool.empty()) {
      auto event = m_pool.top().release();
      m_pool.pop();
      event->reset(stream, enable_timing);
      return MPSEventPtr(event, m_default_deleter);
    }
  }
  auto new_event = std::make_unique<MPSEvent>(++m_event_counter, stream, enable_timing);
  return MPSEventPtr(new_event.release(), m_default_deleter);
}

void MPSEventPool::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  while (!m_pool.empty()) {
    m_pool.pop();
  }
}

id_t MPSEventPool::acquireEvent(bool enable_timing) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEventPtr event = acquireEvent(enable_timing, nullptr);
  TORCH_INTERNAL_ASSERT(event);
  id_t event_id = event->getID();
  m_in_use_events.emplace(event_id, std::move(event));
  return event_id;
}

void MPSEventPool::releaseEvent(id_t event_id) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  // returns the event back to the MPSEventPool
  m_in_use_events.erase(event_id);
}

void MPSEventPool::recordEvent(id_t event_id, bool syncEvent) {
  MPSEvent* event = getInUseEvent(event_id);
  event->record(/*needsLock*/ true, syncEvent);
}

void MPSEventPool::waitForEvent(id_t event_id, bool syncEvent) {
  MPSEvent* event = getInUseEvent(event_id);
  event->wait(/*needsLock*/ true, syncEvent);
}

void MPSEventPool::synchronizeEvent(id_t event_id) {
  MPSEvent* event = getInUseEvent(event_id);
  event->synchronize();
}

bool MPSEventPool::queryEvent(id_t event_id) {
  MPSEvent* event = getInUseEvent(event_id);
  return event->query();
}

double MPSEventPool::elapsedTime(id_t start_event_id, id_t end_event_id) {
  // first make sure notifyListeners are called to capture events' completion times
  dispatch_sync(m_default_stream->queue(), ^() {
    m_default_stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  MPSEvent* start_event = getInUseEvent(start_event_id, false);
  MPSEvent* end_event = getInUseEvent(end_event_id, false);
  // the completion-time notify blocks run on a separate thread, so wait for both
  // events (not just the end one) before reading their timestamps; otherwise the
  // start event's completion time can still be stale when it is read below.
  start_event->waitForCpuSync();
  end_event->waitForCpuSync();
  // GPU timestamps (ns) of the events' command buffers. Use the start event's
  // buffer GPUStartTime and the end event's buffer GPUEndTime, so the interval
  // spans the work between them: when both events share one command buffer (the
  // inductor benchmark pattern: start.record(); kernel(); end.record(); commit)
  // this is exactly that buffer's GPU execution duration, and when they are on
  // different buffers (CUDA-style) it spans from the first's start to the last's
  // end. Reading both events' GPUEndTime instead would give 0 for the shared-
  // buffer case.
  const uint64_t start_time = start_event->getStartTime();
  const uint64_t end_time = end_event->getCompletionTime();

  TORCH_CHECK(start_time > 0 && end_time > 0, "Events were not created with argument 'enable_timing=True'");
  // A region below the GPU timer resolution can still measure end <= start;
  // treat that as 0ms rather than erroring out an entire autotuning choice.
  if (end_time <= start_time) {
    return 0.0;
  }
  return double(end_time - start_time) * 1e-6;
}

MPSEvent* MPSEventPool::getInUseEvent(id_t event_id, bool locked) {
  if (locked) {
    m_mutex.lock();
  }
  TORCH_CHECK(m_in_use_events.count(event_id) > 0, "Invalid Event ID: ", event_id);
  MPSEvent* event = m_in_use_events[event_id].get();
  if (locked) {
    m_mutex.unlock();
  }
  return event;
}

std::shared_ptr<MPSEventPool> getMPSEventPool() {
  static std::shared_ptr<MPSEventPool> event_pool = std::make_shared<MPSEventPool>(getDefaultMPSStream());
  return event_pool;
}

} // namespace at::mps
