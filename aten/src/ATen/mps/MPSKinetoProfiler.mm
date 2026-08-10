//  Copyright © 2026 Apple Inc.

#include <ATen/mps/MPSKinetoProfiler.h>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace at::mps {

#ifdef USE_KINETO

namespace {

using libkineto::ActivityType;
using libkineto::CpuTraceBuffer;
using libkineto::GenericTraceActivity;

// Metal has no per-device queue id to report, and every command buffer this
// records belongs to the single MPS device.
constexpr int64_t kMPSDeviceId = 0;
constexpr int64_t kMPSQueueId = 0;

const std::set<ActivityType>& mpsActivityTypes() {
  static const std::set<ActivityType> types{
      ActivityType::CONCURRENT_KERNEL,
      ActivityType::GPU_MEMCPY,
      ActivityType::GPU_MEMSET,
      ActivityType::GPU_USER_ANNOTATION,
      ActivityType::MPS_RUNTIME,
  };
  return types;
}

// Set while a session is RECORDING. The Metal completion handler runs on every
// dispatch, so it tests this before doing any work.
std::atomic<bool> g_recording{false};

std::mutex g_mutex;
std::vector<GenericTraceActivity> g_activities;

class MPSProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  void start() override {
    {
      std::lock_guard<std::mutex> guard(g_mutex);
      g_activities.clear();
    }
    g_recording = true;
    status_ = libkineto::TraceStatus::RECORDING;
  }

  void stop() override {
    g_recording = false;
    status_ = libkineto::TraceStatus::PROCESSING;
  }

  std::vector<std::string> errors() override {
    return {};
  }

  void processTrace(libkineto::ActivityLogger& logger) override {
    std::lock_guard<std::mutex> guard(g_mutex);
    for (const auto& activity : g_activities) {
      activity.log(logger);
    }
  }

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return std::make_unique<libkineto::DeviceInfo>(libkineto::DeviceInfo{kMPSDeviceId, kMPSDeviceId, "MPS", "MPS"});
  }

  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {libkineto::ResourceInfo{kMPSQueueId, kMPSQueueId, kMPSDeviceId, "Metal command queue"}};
  }

  std::unique_ptr<CpuTraceBuffer> getTraceBuffer() override {
    auto buffer = std::make_unique<CpuTraceBuffer>();
    std::lock_guard<std::mutex> guard(g_mutex);
    for (auto& activity : g_activities) {
      buffer->emplace_activity(activity);
    }
    buffer->gpuOpCount = static_cast<int>(g_activities.size());
    g_activities.clear();
    return buffer;
  }
};

class MPSActivityProfiler : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override {
    static const std::string kName{"mps_profiler"};
    return kName;
  }

  const std::set<ActivityType>& availableActivities() const override {
    return mpsActivityTypes();
  }

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(const std::set<ActivityType>& activity_types,
                                                                 const libkineto::Config& /*config*/) override {
    for (const auto type : activity_types) {
      if (mpsActivityTypes().count(type)) {
        return std::make_unique<MPSProfilerSession>();
      }
    }
    return nullptr;
  }

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(int64_t /*ts_ms*/,
                                                                 int64_t /*duration_ms*/,
                                                                 const std::set<ActivityType>& activity_types,
                                                                 const libkineto::Config& config) override {
    return configure(activity_types, config);
  }
};

} // namespace

bool isMPSKinetoProfilingActive() {
  return g_recording.load(std::memory_order_relaxed);
}

void recordMPSKinetoActivity(const std::string& name,
                             uint64_t correlationId,
                             double gpuStartSeconds,
                             double gpuEndSeconds,
                             bool isCopy) {
  if (!isMPSKinetoProfilingActive() || gpuEndSeconds <= gpuStartSeconds) {
    return;
  }
  GenericTraceActivity activity;
  activity.activityType = isCopy ? ActivityType::GPU_MEMCPY : ActivityType::CONCURRENT_KERNEL;
  activity.activityName = name;
  activity.id = static_cast<int32_t>(correlationId);
  activity.device = kMPSDeviceId;
  activity.resource = kMPSQueueId;
  // Kineto timestamps are microseconds; Metal reports seconds.
  activity.startTime = static_cast<int64_t>(gpuStartSeconds * 1e6);
  activity.endTime = static_cast<int64_t>(gpuEndSeconds * 1e6);

  std::lock_guard<std::mutex> guard(g_mutex);
  g_activities.push_back(std::move(activity));
}

void registerMPSKinetoProfiler() {
  static c10::once_flag flag;
  c10::call_once(
      flag, [] { libkineto::api().registerProfilerFactory([] { return std::make_unique<MPSActivityProfiler>(); }); });
}

#else

bool isMPSKinetoProfilingActive() {
  return false;
}

void recordMPSKinetoActivity(const std::string&, uint64_t, double, double, bool) {}

void registerMPSKinetoProfiler() {}

#endif // USE_KINETO

} // namespace at::mps
