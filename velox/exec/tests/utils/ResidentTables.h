/*
 * ResidentTables.h (2026-08-22)
 *
 * In-memory ("resident") table sources for benchmarks: each TableScan is
 * replaced, at driver-build time, by a ResidentValues source operator that
 * emits a process-resident list of RowVector batches, partitioned across the
 * pipeline's drivers (driver d of N emits batches d, d+N, ...). Unlike
 * ValuesNode(parallelizable=true), which REPLICATES the batches on every
 * driver, this partitions them, so the query runs exactly as it would on a
 * scan whose decode cost is zero. The plan is untouched (same node ids,
 * filters stay as FilterNodes via filtersAsNode), which keeps every other
 * adapter (e.g. the cudf one) unaware of the substitution.
 */
#pragma once

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::exec::test {

/// Process-wide registry: plan node id (of the TableScanNode) -> batches.
class ResidentTableRegistry {
 public:
  static ResidentTableRegistry& instance();

  void set(
      const core::PlanNodeId& scanNodeId,
      std::shared_ptr<const std::vector<RowVectorPtr>> batches);
  std::shared_ptr<const std::vector<RowVectorPtr>> get(
      const core::PlanNodeId& scanNodeId) const;
  void clear();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<
      core::PlanNodeId,
      std::shared_ptr<const std::vector<RowVectorPtr>>>
      tables_;
};

class ResidentValues : public SourceOperator {
 public:
  ResidentValues(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::PlanNodePtr& scanNode,
      std::shared_ptr<const std::vector<RowVectorPtr>> batches,
      size_t numDrivers);

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return batches_ == nullptr || next_ >= batches_->size();
  }

  // Drop our reference as soon as the driver closes: tasks are torn down
  // asynchronously, possibly after the pool owning the batches is gone.
  void close() override {
    SourceOperator::close();
    batches_.reset();
  }

 private:
  std::shared_ptr<const std::vector<RowVectorPtr>> batches_;
  size_t next_;
  size_t stride_;
};

/// Registers the DriverAdapter that performs the TableScan -> ResidentValues
/// substitution for every scan node id present in the registry. Idempotent.
void registerResidentTablesAdapter();
void unregisterResidentTablesAdapter();

} // namespace facebook::velox::exec::test
