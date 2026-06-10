/*
 * GpuFixedRowStore.h
 *
 * Device-side fixed-stride row store types (no-strings variant).
 * This header has NO Velox or folly dependencies so it can be
 * included from both .cpp (host) and .cu (device) files.
 */

#pragma once

#include <cstdint>

// ============================================================================
// Per-column metadata (shared between host and device)
// ============================================================================

struct FieldDesc {
  int32_t offset;      // byte offset within a row
  int32_t byte_width;  // >0 for fixed-width types only
};

// ============================================================================
// GpuFixedRowStore — device-side handle (POD, memcpy-safe)
// ============================================================================

struct GpuFixedRowStore {
  uint8_t* row_buffer;          // [num_rows * row_width] bytes
  int32_t row_width;            // constant stride (8-byte aligned)
  int32_t num_rows;
  int32_t num_fields;
  const FieldDesc* fields;      // [num_fields] on device
};
