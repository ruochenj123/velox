/*
 * GpuFixedRowStore.h
 *
 * Device-side fixed-stride row store types (fixed-width + 16B string slots).
 * This header has NO Velox or folly dependencies so it can be
 * included from both .cpp (host) and .cu (device) files.
 */

#pragma once

#include <cstdint>

// ============================================================================
// Per-column metadata (shared between host and device)
// ============================================================================

// Field kinds. Strings (2026-08-21) live in a FIXED 16-byte slot so the row
// stride stays constant; see RowStrSlot below.
enum : int32_t { kFieldFixed = 0, kFieldString = 1 };

struct FieldDesc {
  int32_t offset;      // byte offset within a row
  int32_t byte_width;  // fixed: type width; string: kRowStrSlotBytes (16)
  int32_t kind = kFieldFixed;
};

// ============================================================================
// Out-of-line strings (2026-08-21; EAGER-COMPACTION design)
//
// A string field occupies a 16-byte slot mirroring Velox's StringView:
//   len <= 12 : [u32 len][12 bytes of data]             (inline)
//   len  > 12 : [u32 len][4-byte prefix][u64 heap off]  (out of line)
// The heap is a per-store byte region (GpuFixedRowStore::chars); offsets are
// relative to it. Join gathers COMPACT survivors' bytes into a fresh heap
// (stringHeapLayout + compactStrings); appending stores (concat, build
// accumulate) rebase offsets (rebaseStringOffsets). A pointer-slot variant
// (no compaction, keep-alive lists) was measured ~5% slower at SF100 warm
// (sequential compacted heaps read faster downstream) and reverted --
// see REVIEW-ROUND5.md. Inline slots are byte-identical to a
// FlatVector<StringView> element, so the CPU pack is a 16B memcpy.
// ============================================================================
constexpr int32_t kRowStrSlotBytes = 16;
constexpr uint32_t kRowStrInlineMax = 12;

struct RowStrSlot {
  uint32_t len;
  uint8_t rest[12]; // inline bytes, or prefix[4] + u64 offset (LE)
};
static_assert(sizeof(RowStrSlot) == kRowStrSlotBytes, "slot must be 16B");

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
  // Out-of-line string heap (2026-08-21). nullptr when no string field is
  // out of line. Appended at the end for ABI safety, as above.
  const uint8_t* chars = nullptr;
  int64_t chars_bytes = 0;
};
