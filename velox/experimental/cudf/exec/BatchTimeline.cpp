/*
 * BatchTimeline.cpp
 */

#include "velox/experimental/cudf/exec/BatchTimeline.h"

#include <fstream>
#include <iostream>

namespace facebook::velox::cudf_velox {

void BatchTimeline::dump() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (events_.empty()) return;

  std::ofstream out(outputPath_);
  if (!out.is_open()) {
    std::cerr << "BatchTimeline: cannot open " << outputPath_ << "\n";
    return;
  }
  out << "timestamp_us,driver_id,pipeline_id,operator,event,rows\n";
  for (auto& e : events_) {
    out << e.timestamp_us << ","
        << e.driver_id << ","
        << e.pipeline_id << ","
        << e.operator_name << ","
        << e.event << ","
        << e.rows << "\n";
  }
  out.close();
  std::cerr << "BatchTimeline: wrote " << events_.size()
            << " events to " << outputPath_ << "\n";
}

} // namespace facebook::velox::cudf_velox
