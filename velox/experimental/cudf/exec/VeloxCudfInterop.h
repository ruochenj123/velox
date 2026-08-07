/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/common/memory/Memory.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace facebook::velox::cudf_velox {

cudf::type_id veloxToCudfTypeId(const TypePtr& type);

cudf::data_type veloxToCudfDataType(const TypePtr& type);

namespace with_arrow {

// Optional per-call phase timing for toCudfTable, filled when a non-null
// pointer is passed. Phases: exportToArrow (array + schema, pure CPU),
// the cudf::from_arrow call (device alloc + copy enqueue), and the stream
// synchronize (waits for the H2D copies to complete).
struct ToCudfTiming {
  int64_t exportNanos{0};
  int64_t fromArrowNanos{0};
  int64_t syncNanos{0};
};

// Two overloads rather than one defaulted `timing` parameter: the 5-arg
// symbol is kept intact so previously compiled objects (EnforceSingleRow,
// Hive connector, ...) keep linking without a full rebuild.
std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<std::string> timestampTimeZone = std::nullopt);

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<std::string> timestampTimeZone,
    ToCudfTiming* timing);

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

// Accepts a Velox TypePtr for recursive metadata construction.
facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    const facebook::velox::RowTypePtr& outputType,
    std::string namePrefix,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace with_arrow

} // namespace facebook::velox::cudf_velox
