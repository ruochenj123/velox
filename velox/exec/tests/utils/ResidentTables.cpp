#include "velox/exec/tests/utils/ResidentTables.h"

#include <iostream>

#include "velox/exec/TableScan.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec::test {

namespace {
constexpr const char* kAdapterLabel = "ResidentTables";
bool registered = false;
} // namespace

ResidentTableRegistry& ResidentTableRegistry::instance() {
  static ResidentTableRegistry registry;
  return registry;
}

void ResidentTableRegistry::set(
    const core::PlanNodeId& scanNodeId,
    std::shared_ptr<const std::vector<RowVectorPtr>> batches) {
  std::lock_guard<std::mutex> l(mutex_);
  tables_[scanNodeId] = std::move(batches);
}

std::shared_ptr<const std::vector<RowVectorPtr>> ResidentTableRegistry::get(
    const core::PlanNodeId& scanNodeId) const {
  std::lock_guard<std::mutex> l(mutex_);
  auto it = tables_.find(scanNodeId);
  return it == tables_.end() ? nullptr : it->second;
}

void ResidentTableRegistry::clear() {
  std::lock_guard<std::mutex> l(mutex_);
  tables_.clear();
}

ResidentValues::ResidentValues(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const core::PlanNodePtr& scanNode,
    std::shared_ptr<const std::vector<RowVectorPtr>> batches,
    size_t numDrivers)
    : SourceOperator(
          driverCtx,
          scanNode->outputType(),
          operatorId,
          scanNode->id(),
          "ResidentValues"),
      batches_(std::move(batches)),
      next_(driverCtx->driverId),
      stride_(numDrivers) {}

RowVectorPtr ResidentValues::getOutput() {
  if (batches_ == nullptr || next_ >= batches_->size()) {
    return nullptr;
  }
  auto out = (*batches_)[next_];
  next_ += stride_;
  return out;
}

void registerResidentTablesAdapter() {
  if (registered) {
    return;
  }
  registered = true;
  DriverAdapter adapter{
      kAdapterLabel,
      {},
      [](const DriverFactory& factory, Driver& driver) -> bool {
        auto ops = driver.operators();
        for (size_t i = 0; i < ops.size(); i++) {
          auto* scan = dynamic_cast<TableScan*>(ops[i]);
          if (scan == nullptr) {
            continue;
          }
          auto batches =
              ResidentTableRegistry::instance().get(scan->planNodeId());
          if (batches == nullptr) {
            continue;
          }
          std::cerr << "[resident] driver " << driver.driverCtx()->driverId
                    << " pipeline " << driver.driverCtx()->pipelineId
                    << ": scan node " << scan->planNodeId() << " -> "
                    << batches->size() << " batches, numDrivers "
                    << factory.numDrivers << std::endl;
          core::PlanNodePtr scanNode;
          for (const auto& node : factory.planNodes) {
            if (node->id() == scan->planNodeId()) {
              scanNode = node;
              break;
            }
          }
          VELOX_CHECK_NOT_NULL(scanNode);
          std::vector<std::unique_ptr<Operator>> replacement;
          replacement.push_back(std::make_unique<ResidentValues>(
              scan->operatorId(),
              driver.driverCtx(),
              scanNode,
              batches,
              factory.numDrivers));
          factory.replaceOperators(driver, i, i + 1, std::move(replacement));
        }
        // MUST return false: Velox stops at the FIRST adapter that returns
        // true (LocalPlanner: `if (adapter.adapt(...)) break;`), and the
        // cudf adapter (which returns true) still has to run after us.
        return false;
      }};
  // Register at the FRONT for the same reason: if the cudf adapter runs
  // first, its `return true` would prevent this one from ever running and
  // the split-less TableScans would wait forever.
  DriverFactory::adapters.insert(DriverFactory::adapters.begin(), adapter);
}

void unregisterResidentTablesAdapter() {
  DriverFactory::adapters.erase(
      std::remove_if(
          DriverFactory::adapters.begin(),
          DriverFactory::adapters.end(),
          [](const DriverAdapter& a) { return a.label == kAdapterLabel; }),
      DriverFactory::adapters.end());
  registered = false;
}

} // namespace facebook::velox::exec::test
