#pragma once
// Global flags for operator-timeline tracing (header-only, no link dependency).
// Set by the benchmark binary; read by TableScan and cudf operators.
// Uses thread_local for per-driver state (each driver runs on its own thread).

#include <atomic>
#include <chrono>
#include <cstdio>

namespace facebook::velox::cudf_velox {

/// Master enable flag — set to true by the benchmark before running.
inline std::atomic<bool>& benchmarkTimelineEnabled() {
  static std::atomic<bool> flag{false};
  return flag;
}

/// Per-thread (per-driver) state for scan tracing.
/// scanTraceNeeded: true means the next TableScan::getOutput() should emit
/// "Scan start". Set to true by CudfFromVelox after producing a GPU batch.
/// driverId / pipelineId: cached for fast access.
struct ScanTraceState {
  bool scanTraceNeeded{true};
  int driverId{-1};
  int pipelineId{-1};
};

inline ScanTraceState& threadScanTraceState() {
  static thread_local ScanTraceState state;
  return state;
}

} // namespace facebook::velox::cudf_velox
