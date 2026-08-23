/*
 * DedicatedStream.h (2026-08-21)
 *
 * Per-operator CUDA streams for the row path. cudf's global stream pool has
 * 32 streams handed out round-robin; at 24 drivers x several operators each,
 * unrelated operators alias onto one stream, so a build's synchronize() (or
 * a pinned pack's per-batch sync) also waits for some other driver's queued
 * 1M-row probe/aggregation work -- the nondeterministic multi-second stalls
 * seen in the SF30 sweeps. A dedicated stream per row-path operator removes
 * that false dependency. Streams are kept alive for the process lifetime:
 * RowStoreVectors carry the stream view downstream, and a few hundred
 * streams are cheap.
 */
#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <mutex>
#include <vector>

namespace facebook::velox::cudf_velox {

inline rmm::cuda_stream_view acquireDedicatedStream() {
  static std::mutex mu;
  static std::vector<std::unique_ptr<rmm::cuda_stream>> streams;
  std::lock_guard<std::mutex> g(mu);
  streams.push_back(std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking));
  return streams.back()->view();
}

} // namespace facebook::velox::cudf_velox
