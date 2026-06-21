/*
 * BatchTimeline.h
 *
 * Lightweight per-batch timeline logger for pipeline visualization.
 * Records [timestamp_us, driver_id, operator_name, event, rows] events
 * to a thread-safe global log, then dumps to CSV at end.
 *
 * Enabled via CudfConfig::benchmarkLogTimeline or --log_timeline=true.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

struct TimelineEvent {
  int64_t timestamp_us;  // microseconds since epoch
  int32_t driver_id;
  int32_t pipeline_id;
  const char* operator_name;
  const char* event;     // "start" or "end"
  int64_t rows;          // batch row count (0 if N/A)
};

/// Global singleton timeline logger.
class BatchTimeline {
 public:
  static BatchTimeline& instance() {
    static BatchTimeline inst;
    return inst;
  }

  bool enabled() const { return enabled_; }
  void setEnabled(bool v) { enabled_ = v; }

  void setOutputPath(const std::string& path) { outputPath_ = path; }

  void log(int32_t driverId, int32_t pipelineId,
           const char* opName, const char* event, int64_t rows = 0) {
    if (!enabled_) return;
    auto now = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - epoch_).count();
    std::lock_guard<std::mutex> lk(mu_);
    events_.push_back({us, driverId, pipelineId, opName, event, rows});
  }

  /// Resets the epoch (call at start of benchmark iteration).
  void resetEpoch() {
    epoch_ = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    events_.clear();
  }

  /// Dump events to CSV file.
  void dump() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (events_.empty()) return;
    FILE* f = fopen(outputPath_.c_str(), "w");
    if (!f) {
      fprintf(stderr, "BatchTimeline: cannot open %s\n", outputPath_.c_str());
      return;
    }
    fprintf(f, "timestamp_us,driver_id,pipeline_id,operator,event,rows\n");
    for (auto& e : events_) {
      fprintf(f, "%ld,%d,%d,%s,%s,%ld\n",
          (long)e.timestamp_us, e.driver_id, e.pipeline_id,
          e.operator_name, e.event, (long)e.rows);
    }
    fclose(f);
    fprintf(stderr, "BatchTimeline: wrote %zu events to %s\n",
        events_.size(), outputPath_.c_str());
  }

  /// Get events for in-process analysis.
  std::vector<TimelineEvent> getEvents() const {
    std::lock_guard<std::mutex> lk(mu_);
    return events_;
  }

 private:
  BatchTimeline() = default;

  bool enabled_ = false;
  std::string outputPath_ = "timeline.csv";
  std::chrono::steady_clock::time_point epoch_ =
      std::chrono::steady_clock::now();
  mutable std::mutex mu_;
  std::vector<TimelineEvent> events_;
};

/// RAII helper for timing an operator scope.
class TimelineScope {
 public:
  TimelineScope(int32_t driverId, int32_t pipelineId,
                const char* opName, int64_t rows = 0)
      : driverId_(driverId), pipelineId_(pipelineId),
        opName_(opName), rows_(rows) {
    if (BatchTimeline::instance().enabled()) {
      BatchTimeline::instance().log(driverId, pipelineId, opName, "start", rows);
    }
  }

  ~TimelineScope() {
    if (BatchTimeline::instance().enabled()) {
      BatchTimeline::instance().log(
          driverId_, pipelineId_, opName_, "end", rows_);
    }
  }

  void setRows(int64_t rows) { rows_ = rows; }

 private:
  int32_t driverId_;
  int32_t pipelineId_;
  const char* opName_;
  int64_t rows_;
};

} // namespace facebook::velox::cudf_velox
