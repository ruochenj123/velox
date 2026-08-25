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

  /// Spine pack: rows shipped across the boundary for the chain whose
  /// terminal join is `endpointJoinId` (the denominator).
  void recordPacked(const std::string& endpointJoinId, int64_t rows) {
    std::lock_guard<std::mutex> l(mu_);
    stats_[endpointJoinId].packed += rows;
  }

  /// Terminal probe of the chain: rows leaving the chain (the numerator).
  /// A CPU exit reports 0 -- with direct host emission the deferred payload
  /// never crosses the boundary, so deferral always wins there.
  void recordSurvived(const std::string& endpointJoinId, int64_t rows) {
    std::lock_guard<std::mutex> l(mu_);
    auto& e = stats_[endpointJoinId];
    e.survived += rows;
    e.reports++;
  }

  /// True once the endpoint has reported at least one output batch AND the
  /// cumulative survival ratio is at or below `threshold`. `packed` runs a
  /// few batches ahead of `survived` (pipelining), which only biases the
  /// ratio DOWN -- acceptable for a one-way eager->defer switch.
  bool shouldDefer(const std::string& endpointJoinId, double threshold) const {
    std::lock_guard<std::mutex> l(mu_);
    auto it = stats_.find(endpointJoinId);
    if (it == stats_.end() || it->second.reports == 0 ||
        it->second.packed == 0) {
      return false; // no endpoint observation yet: eager
    }
    return static_cast<double>(it->second.survived) <=
        threshold * static_cast<double>(it->second.packed);
  }

 private:
  struct Entry {
    int64_t packed{0};
    int64_t survived{0};
    int64_t reports{0};
  };
  mutable std::mutex mu_;
  std::unordered_map<std::string, Entry> stats_;
};

} // namespace facebook::velox::cudf_velox
