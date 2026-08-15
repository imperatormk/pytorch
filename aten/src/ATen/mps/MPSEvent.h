//  Copyright © 2023 Apple Inc.

#pragma once

#include <ATen/mps/MPSStream.h>
#include <algorithm>
#include <condition_variable>
#include <ctime>
#include <memory>
#include <mutex>
#include <stack>
#include <vector>

namespace at::mps {

// Shared timing + CPU-sync state for a single recorded event. Held by a
// shared_ptr so a command-buffer completion handler that writes the GPU
// timestamps can outlive the owning MPSEvent without a use-after-free: the
// handler fires asynchronously on a GPU dispatch thread and may run after the
// MPSEvent (e.g. a torch.mps.Event) has already been destroyed or returned to
// the pool. Both the MPSEvent and the captured block keep a strong reference,
// so the writes always land in a live object.
struct MPSEventCpuSync {
  std::mutex mutex{};
  std::condition_variable cv{};
  bool completed = false;
  uint64_t completion_time = 0;
  uint64_t start_time = 0;
};

// NOTE: don't create instances of this class directly.
// Use MPSEventPool to acquire instances of MPSEvent.
class MPSEvent {
 public:
  explicit MPSEvent(id_t ID, MPSStream* stream, bool enable_timing);
  ~MPSEvent();

  // records an event on the stream
  void record(bool needsLock, bool syncEvent = false);
  // makes all future work submitted to the stream wait for this event.
  bool wait(bool needsLock, bool syncEvent = false);
  // schedules a notifyListener callback for the event.
  bool notify(bool needsLock, MTLSharedEventNotificationBlock block);
  // checks if events are already signaled.
  bool query() const;
  // blocks the CPU thread until all the GPU work that were scheduled
  // prior to recording this event are completed.
  bool synchronize();
  // resets this event with new parameters in case it gets reused from the event
  // pool
  void reset(MPSStream* stream, bool enable_timing);
  // returns the unique ID of the event instance
  id_t getID() const {
    return m_id;
  }
  // returns the stream this event is bound to (set at acquire time)
  MPSStream* stream() const {
    return m_stream;
  }
  // returns the completion timestamp of the event
  uint64_t getCompletionTime() const {
    return m_cpu_sync->completion_time;
  }
  // returns the GPU start timestamp of the event's command buffer
  uint64_t getStartTime() const {
    return m_cpu_sync->start_time;
  }
  // if already recorded, waits for cpu_sync_cv to be signaled
  void waitForCpuSync();

 private:
  id_t m_id;
  // enables measuring the completion time of the notifyListener of this event
  bool m_enable_timing;
  uint64_t m_signalCounter = 0;
  MPSStream* m_stream = nullptr;
  MTLSharedEvent_t m_event = nullptr;
  MTLSharedEventListener* m_listener = nullptr;
  // CPU-sync + timing state, shared with the command-buffer completion handler
  // so it can be written safely even if this MPSEvent is destroyed first.
  std::shared_ptr<MPSEventCpuSync> m_cpu_sync = std::make_shared<MPSEventCpuSync>();

  void recordLocked(bool syncEvent);
  bool waitLocked(bool syncEvent);
  bool notifyLocked(MTLSharedEventNotificationBlock block);
  void notifyCpuSync();
  static uint64_t getTime() {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
  }
};

typedef std::unique_ptr<MPSEvent, std::function<void(MPSEvent*)>> MPSEventPtr;

class MPSEventPool {
 public:
  explicit MPSEventPool(MPSStream* default_stream);
  ~MPSEventPool();

  MPSEventPtr acquireEvent(bool enable_timing, MPSStream* stream);
  void emptyCache();

  // these are mainly used for MPSHooks and torch.mps.Event() bindings
  id_t acquireEvent(bool enable_timing);
  void releaseEvent(id_t event_id);
  void recordEvent(id_t event_id, bool syncEvent);
  void waitForEvent(id_t event_id, bool syncEvent);
  void synchronizeEvent(id_t event_id);
  bool queryEvent(id_t event_id);
  // returns elapsed time between two recorded events in milliseconds
  double elapsedTime(id_t start_event_id, id_t end_event_id);

 private:
  MPSStream* m_default_stream = nullptr;
  std::recursive_mutex m_mutex;
  std::stack<std::unique_ptr<MPSEvent>> m_pool{};
  // dictionary to associate event IDs with event objects
  // used to retain in-use events out of the pool
  // for torch.mps.Event() bindings.
  std::unordered_map<id_t, MPSEventPtr> m_in_use_events{};
  uint64_t m_event_counter = 0;
  std::function<void(MPSEvent*)> m_default_deleter;

  MPSEvent* getInUseEvent(id_t event_id, bool locked = true);
};

// shared_ptr is used to get MPSEventPool destroyed after dependent instances
std::shared_ptr<MPSEventPool> getMPSEventPool();

} // namespace at::mps
