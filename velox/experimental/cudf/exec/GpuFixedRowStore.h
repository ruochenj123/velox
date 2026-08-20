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

/// Per-field mapping for selective gather: maps a field from a source row
/// to a position in an output row.
struct FieldMapping {
  int32_t src_offset;  // byte offset within source row
  int32_t dst_offset;  // byte offset within output row
  int32_t byte_width;  // field width in bytes
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
  // Optional null sidecar (2026-08-17 null support): bit SET = NULL, so an
  // absent/zeroed sidecar means all-valid. Addressing:
  //   null_bytes[row * null_stride + (field >> 3)] & (1 << (field & 7)).
  // Defaults keep every legacy construction site null-free. Appended at the
  // struct end for partial-rebuild ABI safety (see REVIEW-GUIDE).
  const uint8_t* null_bytes = nullptr;
  int32_t null_stride = 0;
};
