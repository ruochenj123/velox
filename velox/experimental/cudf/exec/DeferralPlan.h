/*
 * DeferralPlan.h
 *
 * Plan-level column classification for WHOLE-QUERY payload deferral (see
 * DESIGN-whole-query-deferral.md §2). Given the plan and the id of the
 * HashJoinNode adjacent to a conversion boundary (or of a join whose output
 * is being emitted), collect the join-key names of the adjacent join and of
 * every ANCESTOR join up to the first provenance boundary -- the first
 * ancestor that is not a HashJoinNode (aggregation, sort, limit, project,
 * filter, output). Everything a downstream join hashes on must cross the
 * boundary as a value; everything else flows as a row reference.
 *
 * Both sides' keys of every join are returned; callers intersect with their
 * own input type (column names are unique across a Velox join's output, so
 * the intersection is exact).
 *
 * Header-only; no CUDA dependencies.
 */

#pragma once

#include "velox/core/PlanNode.h"

#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

struct ChainKeyNames {
  /// Keys of the join identified by joinNodeId (left then right).
  std::vector<std::string> adjacent;
  /// Keys of the ancestor joins in the chain (nearest first, left+right).
  std::vector<std::string> later;
  /// Number of ancestor joins in the chain (0 == the adjacent join is the
  /// terminal row join of its pipeline).
  int32_t chainLength{0};
};

namespace deferral_detail {
inline bool findPath(
    const core::PlanNodePtr& node,
    const std::string& id,
    std::vector<core::PlanNodePtr>& path) {
  if (node == nullptr) {
    return false;
  }
  path.push_back(node);
  if (node->id() == id) {
    return true;
  }
  for (const auto& src : node->sources()) {
    if (findPath(src, id, path)) {
      return true;
    }
  }
  path.pop_back();
  return false;
}

inline void appendKeys(
    const std::shared_ptr<const core::HashJoinNode>& join,
    std::vector<std::string>& out) {
  for (const auto& k : join->leftKeys()) {
    out.push_back(k->name());
  }
  for (const auto& k : join->rightKeys()) {
    out.push_back(k->name());
  }
}
} // namespace deferral_detail

/// Collect adjacent + later join keys for `joinNodeId` under `root`.
/// Returns an empty result (chainLength 0, no keys) if the id is not found.
inline ChainKeyNames collectChainKeyNames(
    const core::PlanNodePtr& root,
    const std::string& joinNodeId) {
  ChainKeyNames out;
  std::vector<core::PlanNodePtr> path; // root ... joinNode
  if (!deferral_detail::findPath(root, joinNodeId, path) || path.empty()) {
    return out;
  }
  auto adjacent =
      std::dynamic_pointer_cast<const core::HashJoinNode>(path.back());
  if (adjacent == nullptr) {
    return out;
  }
  deferral_detail::appendKeys(adjacent, out.adjacent);
  // Walk ancestors nearest-first; the chain ends at the first non-join.
  for (int32_t i = static_cast<int32_t>(path.size()) - 2; i >= 0; i--) {
    auto join = std::dynamic_pointer_cast<const core::HashJoinNode>(path[i]);
    if (join == nullptr) {
      break;
    }
    deferral_detail::appendKeys(join, out.later);
    out.chainLength++;
  }
  return out;
}

} // namespace facebook::velox::cudf_velox
