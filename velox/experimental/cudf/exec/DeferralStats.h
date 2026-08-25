/*
 * DeferralStats.h (2026-08-24)
 *
 * Process-wide observed join reduction, keyed by plan node id. Probes
 * record (probedRows, matchedRows) every batch; the boundary pack consults
 * it to decide defer-vs-eager ADAPTIVELY: eager by default (first
 * execution observes), deferred once a prior run of the same plan node
 * measured reduction below the threshold. Matches the warm-repeat
 * benchmark protocol (repeat 0 = observation run) and, in a long-running
 * engine, ordinary steady-state re-execution.
 */
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

class DeferralStats {
 public:
  static DeferralStats& instance() {
    static DeferralStats s;
    return s;
  }

  void record(const std::string& joinNodeId, int64_t probed, int64_t matched) {
    std::lock_guard<std::mutex> l(mu_);
    auto& e = stats_[joinNodeId];
    e.first += probed;
    e.second += matched;
  }

  /// True once observations exist AND the measured match rate is at or
  /// below `threshold` (i.e. the chain eliminates enough for the deferred
  /// round trip to win).
  bool shouldDefer(const std::string& joinNodeId, double threshold) const {
    std::lock_guard<std::mutex> l(mu_);
    auto it = stats_.find(joinNodeId);
    if (it == stats_.end() || it->second.first == 0) {
      return false; // no observation yet: eager
    }
    return static_cast<double>(it->second.second) /
        static_cast<double>(it->second.first) <=
        threshold;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::pair<int64_t, int64_t>> stats_;
};

} // namespace facebook::velox::cudf_velox
