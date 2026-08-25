/*
 * Lazily materializable result vector (2026-08-25).
 *
 * A vector that carries its rows in a non-columnar native layout (e.g. the
 * GPU row layout D2H'd at a CPU exit) implements this interface so generic
 * consumers that need Velox columns (result printing, checksums) can ask for
 * them explicitly. Everything else (sinks, stats) uses size() only.
 */
#pragma once

#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

class MaterializableVector {
 public:
  virtual ~MaterializableVector() = default;
  /// Full extraction into ordinary Velox columns. Called on demand only.
  virtual RowVectorPtr materialize() const = 0;
};

} // namespace facebook::velox
