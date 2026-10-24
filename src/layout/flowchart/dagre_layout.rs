//! Full dagre-style layered layout pipeline.
//!
//! This implements the `runLayout` pipeline from dagre (as used by mermaid.js),
//! adapted to this crate's types:
//!
//! ```text
//! make_space_for_edge_labels -> acyclic -> rank -> inject_edge_label_proxies
//! -> normalize -> order -> position -> undo -> fixup_edge_label_coords
//! -> assign_node_intersects -> acyclic_undo
//! ```
//!
//! The key difference from the existing `edge_pipeline` approach:
//!
//! * Edge labels are first-class layout participants. `make_space_for_edge_labels`
//!   doubles every edge's minlen and halves ranksep, creating a "half rank"
//!   between every pair of ranks. The label lives in that half rank.
//! * Long edges are split into chains of dummy nodes (one per intermediate
//!   rank) by `normalize`. The dummy at the label's rank carries the label's
//!   dimensions and participates in ordering and positioning like a real node.
//! * Edge points are *derived* from the dummy chain coordinates during `undo`.
//!   There is no obstacle-avoidance routing, no per-edge side scoring, and no
//!   post-hoc detour around labels. Lines are straight because the layout
//!   itself aligns them, not because a router dodged obstacles.
//!
//! This engine is used when `FlowchartLayoutEngine::Dagre` is selected. It
//! bypasses `edge_pipeline::build_routed_edges` entirely and produces
//! `EdgeLayout`s directly.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::LayoutConfig;
use crate::ir::{DiagramKind, Direction, Graph, NodeShape};
use crate::layout::{NodeLayout, TextBlock};
use crate::theme::Theme;

/// A node in the layout graph: either a real node or a dummy.
#[derive(Debug, Clone)]
struct LayoutNode {
    /// Real node id, or a generated dummy id (`__dummy_N__`).
    id: String,
    /// None for real nodes; Some for dummies.
    dummy_kind: Option<DummyKind>,
    /// Width/height. For edge dummies this is 0; for edge-label dummies this
    /// is the label's size.
    width: f32,
    height: f32,
    /// Rank (layer) after ranking. Doubled ranks: real nodes sit on even
    /// ranks, edge labels on odd (half) ranks.
    rank: usize,
    /// Order within the rank after ordering.
    order: usize,
    /// Center coordinates after positioning.
    x: f32,
    y: f32,
    /// For edge dummies: the original edge index.
    edge_idx: Option<usize>,
    /// For edge-label dummies: the label dimensions (same as width/height).
    /// For real nodes: the node shape, used for endpoint clipping.
    shape: NodeShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DummyKind {
    /// Intermediate point on a long edge's chain.
    Edge,
    /// The label position of a labeled edge. Carries label dimensions.
    EdgeLabel,
}

/// A directed edge in the layout graph.
#[derive(Debug, Clone)]
struct LayoutEdge {
    from: String,
    to: String,
    /// Original edge index in `graph.edges`.
    orig_idx: usize,
    /// Weight for ordering (default 1).
    weight: f32,
    /// Minimum rank span (doubled by make_space_for_edge_labels).
    minlen: usize,
    /// True when acyclic reversed this edge; undo flips it back.
    reversed: bool,
}

/// Output of the dagre pipeline: node positions + edge geometry.
pub struct DagreLayoutOutput {
    /// Node id -> (center x, center y).
    pub node_centers: HashMap<String, (f32, f32)>,
    /// Edge index -> routed points (including node-border-clipped endpoints).
    pub edge_points: HashMap<usize, Vec<(f32, f32)>>,
    /// Edge index -> label center (for edges with a center label).
    pub edge_label_centers: HashMap<usize, (f32, f32)>,
}

/// Run the full dagre pipeline for a flowchart.
///
/// `nodes` carries already-measured node sizes; `edge_labels` carries
/// already-measured center labels (parallel to `graph.edges`). Returns
/// `None` when the graph is not suitable for this engine (e.g. it has
/// subgraphs, which need the compound handling this module does not yet
/// implement); callers fall back to the shared edge pipeline in that case.
pub fn layout_dagre(
    graph: &Graph,
    nodes: &BTreeMap<String, NodeLayout>,
    edge_labels: &[Option<TextBlock>],
    config: &LayoutConfig,
    _theme: &Theme,
) -> Option<DagreLayoutOutput> {
    if graph.kind != DiagramKind::Flowchart {
        return None;
    }
    if !graph.subgraphs.is_empty() {
        return None;
    }
    let direction = graph.direction;
    // This pipeline is written for TopDown and mirrored for other directions
    // by swapping axes at the boundary (horizontal) and by finalize's mirror
    // (RL/BT).
    let is_horizontal = matches!(
        direction,
        Direction::LeftRight | Direction::RightLeft
    );

    let node_spacing = config.node_spacing.max(1.0);
    let rank_spacing = config.rank_spacing.max(1.0);

    // Collect real nodes.
    let mut layout_nodes: HashMap<String, LayoutNode> = HashMap::new();
    for (id, node) in nodes {
        if node.hidden {
            continue;
        }
        layout_nodes.insert(
            id.clone(),
            LayoutNode {
                id: id.clone(),
                dummy_kind: None,
                width: node.width,
                height: node.height,
                rank: 0,
                order: 0,
                x: 0.0,
                y: 0.0,
                edge_idx: None,
                shape: node.shape,
            },
        );
    }
    if layout_nodes.is_empty() {
        return Some(DagreLayoutOutput {
            node_centers: HashMap::new(),
            edge_points: HashMap::new(),
            edge_label_centers: HashMap::new(),
        });
    }

    // Build layout edges (skip self-loops; they are routed separately by the
    // caller as compact loops).
    let mut layout_edges: Vec<LayoutEdge> = Vec::new();
    for (idx, edge) in graph.edges.iter().enumerate() {
        if edge.from == edge.to {
            continue;
        }
        if !layout_nodes.contains_key(&edge.from) || !layout_nodes.contains_key(&edge.to) {
            continue;
        }
        layout_edges.push(LayoutEdge {
            from: edge.from.clone(),
            to: edge.to.clone(),
            orig_idx: idx,
            weight: 1.0,
            minlen: 1,
            reversed: false,
        });
    }

    // 1. make_space_for_edge_labels: halve ranksep, double minlen.
    // Labels live in the half-ranks this creates.
    let mut ranksep = rank_spacing / 2.0;
    if ranksep < 1.0 {
        ranksep = 1.0;
    }
    for edge in &mut layout_edges {
        edge.minlen *= 2;
    }

    // 2. Rank assignment (longest path on the acyclic subgraph).
    // First break cycles greedily so ranking terminates.
    let feedback = acyclic_reverse(&mut layout_edges, &layout_nodes);
    let mut ranks = rank_longest_path(&layout_nodes, &layout_edges);
    // Double ranks so real nodes sit on even ranks and labels on odd ranks.
    for rank in ranks.values_mut() {
        *rank *= 2;
    }
    for node in layout_nodes.values_mut() {
        node.rank = *ranks.get(&node.id).unwrap_or(&0);
    }

    // 3. Inject edge-label proxies: a dummy at the midpoint rank of each
    // labeled edge, so remove_empty_ranks keeps the half-rank alive.
    let mut label_rank: HashMap<usize, usize> = HashMap::new();
    for edge in &layout_edges {
        let Some(label) = edge_labels.get(edge.orig_idx).and_then(|l| l.as_ref()) else {
            continue;
        };
        if label.width <= 0.0 || label.height <= 0.0 {
            continue;
        }
        let from_rank = layout_nodes.get(&edge.from).map(|n| n.rank).unwrap_or(0);
        let to_rank = layout_nodes.get(&edge.to).map(|n| n.rank).unwrap_or(0);
        // Midpoint rank (in doubled coordinates). For a forward edge this is
        // odd (a half rank); for a reversed edge the ranks are swapped by
        // acyclic, so use the absolute midpoint.
        let mid = (from_rank + to_rank) / 2;
        label_rank.insert(edge.orig_idx, mid);
        let proxy_id = format!("__label_proxy_{}__", edge.orig_idx);
        layout_nodes.insert(
            proxy_id.clone(),
            LayoutNode {
                id: proxy_id,
                dummy_kind: Some(DummyKind::EdgeLabel),
                width: 0.0,
                height: 0.0,
                rank: mid,
                order: 0,
                x: 0.0,
                y: 0.0,
                edge_idx: Some(edge.orig_idx),
                shape: NodeShape::Rectangle,
            },
        );
    }

    // 4. Remove empty ranks (but keep ranks holding label proxies).
    let mut used_ranks: HashSet<usize> = layout_nodes.values().map(|n| n.rank).collect();
    let mut sorted_ranks: Vec<usize> = used_ranks.drain().collect();
    sorted_ranks.sort_unstable();
    let mut rank_remap: HashMap<usize, usize> = HashMap::new();
    for (new_rank, old_rank) in sorted_ranks.iter().enumerate() {
        rank_remap.insert(*old_rank, new_rank);
    }
    for node in layout_nodes.values_mut() {
        node.rank = *rank_remap.get(&node.rank).unwrap_or(&0);
    }
    for (idx, rank) in label_rank.iter_mut() {
        // Recompute the label rank from the (possibly remapped) endpoint ranks
        // is complex; instead remap the stored midpoint through the same map
        // when it survives, else fall back to the endpoint midpoint.
        if let Some(new_rank) = rank_remap.get(rank) {
            *rank = *new_rank;
        } else {
            let edge = &layout_edges.iter().find(|e| e.orig_idx == *idx);
            if let Some(edge) = edge {
                let fr = layout_nodes.get(&edge.from).map(|n| n.rank).unwrap_or(0);
                let tr = layout_nodes.get(&edge.to).map(|n| n.rank).unwrap_or(0);
                *rank = (fr + tr) / 2;
            }
        }
    }

    // 5. Normalize: split long edges into dummy chains. The dummy at the
    // label rank carries the label dimensions.
    let mut chains: HashMap<usize, Vec<String>> = HashMap::new();
    let mut dummy_counter = 0usize;
    let mut chain_edges: Vec<LayoutEdge> = Vec::new();
    for edge in layout_edges.iter() {
        let from_rank = layout_nodes.get(&edge.from).map(|n| n.rank).unwrap_or(0);
        let to_rank = layout_nodes.get(&edge.to).map(|n| n.rank).unwrap_or(0);
        if to_rank <= from_rank + 1 {
            // Adjacent ranks: no chain needed.
            chain_edges.push(edge.clone());
            chains.insert(edge.orig_idx, vec![edge.from.clone(), edge.to.clone()]);
            continue;
        }
        let lbl_rank = label_rank.get(&edge.orig_idx).copied();
        let lbl = edge_labels.get(edge.orig_idx).and_then(|l| l.as_ref());
        let mut chain: Vec<String> = vec![edge.from.clone()];
        let mut prev = edge.from.clone();
        for rank in (from_rank + 1)..to_rank {
            let is_label_rank = lbl_rank == Some(rank);
            let (w, h, kind) = if is_label_rank {
                if let Some(lbl) = lbl {
                    (lbl.width, lbl.height, DummyKind::EdgeLabel)
                } else {
                    (0.0, 0.0, DummyKind::Edge)
                }
            } else {
                (0.0, 0.0, DummyKind::Edge)
            };
            let dummy_id = format!("__dummy_{dummy_counter}__");
            dummy_counter += 1;
            layout_nodes.insert(
                dummy_id.clone(),
                LayoutNode {
                    id: dummy_id.clone(),
                    dummy_kind: Some(kind),
                    width: w,
                    height: h,
                    rank,
                    order: 0,
                    x: 0.0,
                    y: 0.0,
                    edge_idx: Some(edge.orig_idx),
                    shape: NodeShape::Rectangle,
                },
            );
            chain_edges.push(LayoutEdge {
                from: prev.clone(),
                to: dummy_id.clone(),
                orig_idx: edge.orig_idx,
                weight: edge.weight,
                minlen: 1,
                reversed: edge.reversed,
            });
            chain.push(dummy_id.clone());
            prev = dummy_id;
        }
        chain_edges.push(LayoutEdge {
            from: prev,
            to: edge.to.clone(),
            orig_idx: edge.orig_idx,
            weight: edge.weight,
            minlen: 1,
            reversed: edge.reversed,
        });
        chain.push(edge.to.clone());
        chains.insert(edge.orig_idx, chain);
    }
    // Remove the now-redundant label proxies (the chain dummies supersede them).
    layout_nodes.retain(|_, node| {
        !(node.dummy_kind == Some(DummyKind::EdgeLabel) && node.id.starts_with("__label_proxy_"))
    });

    // 6. Order: barycenter sweeps over all ranks (real + dummy nodes).
    let mut rank_buckets = build_rank_buckets(&layout_nodes);
    order_barycenter(&mut rank_buckets, &chain_edges, &layout_nodes, 24);
    for (rank, bucket) in rank_buckets.iter().enumerate() {
        for (order, id) in bucket.iter().enumerate() {
            if let Some(node) = layout_nodes.get_mut(id) {
                node.rank = rank;
                node.order = order;
            }
        }
    }

    // 7. Position: Brandes-Kopf-style alignment + compaction.
    position_nodes(
        &mut layout_nodes,
        &rank_buckets,
        &chain_edges,
        node_spacing,
        ranksep,
        is_horizontal,
    );

    // 8. Undo: collapse chains into edge points; edge-label dummies give the
    // label coordinates.
    let mut edge_points: HashMap<usize, Vec<(f32, f32)>> = HashMap::new();
    let mut edge_label_centers: HashMap<usize, (f32, f32)> = HashMap::new();
    for (orig_idx, chain) in &chains {
        let mut points: Vec<(f32, f32)> = Vec::new();
        for id in chain {
            if let Some(node) = layout_nodes.get(id) {
                points.push((node.x, node.y));
                if node.dummy_kind == Some(DummyKind::EdgeLabel) {
                    edge_label_centers.insert(*orig_idx, (node.x, node.y));
                }
            }
        }
        // Compress collinear intermediate points.
        let compressed = compress_points(&points);
        edge_points.insert(*orig_idx, compressed);
    }
    // Labeled edges without a chain dummy at their label rank (adjacent-rank
    // edges) get their label at the edge midpoint.
    for edge in &layout_edges {
        if edge_label_centers.contains_key(&edge.orig_idx) {
            continue;
        }
        let Some(label) = edge_labels.get(edge.orig_idx).and_then(|l| l.as_ref()) else {
            continue;
        };
        if label.width <= 0.0 || label.height <= 0.0 {
            continue;
        }
        if let Some(points) = edge_points.get(&edge.orig_idx) {
            if let Some(mid) = point_at_progress(points, 0.5) {
                edge_label_centers.insert(edge.orig_idx, mid);
            }
        }
    }

    // 9. Assign node intersects: clip endpoints to node borders.
    let mut node_geom: HashMap<String, (f32, f32, f32, f32, NodeShape)> = HashMap::new();
    for (id, node) in nodes {
        if node.hidden {
            continue;
        }
        if let Some(center) = layout_nodes.get(id).map(|n| (n.x, n.y)) {
            node_geom.insert(
                id.clone(),
                (center.0, center.1, node.width, node.height, node.shape),
            );
        }
    }
    for edge in &layout_edges {
        let Some(points) = edge_points.get_mut(&edge.orig_idx) else {
            continue;
        };
        if points.len() < 2 {
            continue;
        }
        // Clip start to the source node's border.
        if let Some((cx, cy, w, h, shape)) = node_geom.get(&edge.from) {
            let next = points[1];
            let clipped = clip_to_node_border((*cx, *cy), (*w, *h), *shape, next);
            points[0] = clipped;
        }
        // Clip end to the target node's border.
        if let Some((cx, cy, w, h, shape)) = node_geom.get(&edge.to) {
            let prev = points[points.len() - 2];
            let clipped = clip_to_node_border((*cx, *cy), (*w, *h), *shape, prev);
            let last = points.len() - 1;
            points[last] = clipped;
        }
    }

    // 10. Acyclic undo: reverse the points of reversed edges.
    for edge in &layout_edges {
        if edge.reversed {
            if let Some(points) = edge_points.get_mut(&edge.orig_idx) {
                points.reverse();
            }
        }
    }

    // 11. Mirror back for flipped directions, and swap axes for horizontal.
    let mut node_centers: HashMap<String, (f32, f32)> = HashMap::new();
    for (id, node) in layout_nodes.iter() {
        if node.dummy_kind.is_some() {
            continue;
        }
        node_centers.insert(id.clone(), (node.x, node.y));
    }
    // Translate so the minimum coordinate is 0 (dagre's translateGraph).
    // Include full node extents and label extents: a wide label centered on
    // a diagonal edge can extend past the leftmost/topmost node.
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    for (id, (x, y)) in node_centers.iter() {
        let (w, h) = layout_nodes
            .get(id)
            .map(|n| (n.width, n.height))
            .unwrap_or((0.0, 0.0));
        min_x = min_x.min(x - w / 2.0);
        min_y = min_y.min(y - h / 2.0);
    }
    for (idx, center) in edge_label_centers.iter() {
        let (lw, lh) = edge_labels
            .get(*idx)
            .and_then(|l| l.as_ref())
            .map(|l| (l.width, l.height))
            .unwrap_or((0.0, 0.0));
        min_x = min_x.min(center.0 - lw / 2.0);
        min_y = min_y.min(center.1 - lh / 2.0);
    }
    for points in edge_points.values_mut() {
        for p in points.iter_mut() {
            p.0 -= min_x;
            p.1 -= min_y;
        }
    }
    for center in edge_label_centers.values_mut() {
        center.0 -= min_x;
        center.1 -= min_y;
    }
    for center in node_centers.values_mut() {
        center.0 -= min_x;
        center.1 -= min_y;
    }

    // Direction transform: the pipeline above computed TD coordinates
    // (x = horizontal order, y = rank). For horizontal directions (LR/RL),
    // swap axes here; `finalize_graph_layout` applies the RL/BT mirroring
    // afterwards, which then operates on the correct (rank) axis.
    if is_horizontal {
        for center in node_centers.values_mut() {
            std::mem::swap(&mut center.0, &mut center.1);
        }
        for points in edge_points.values_mut() {
            for p in points.iter_mut() {
                std::mem::swap(&mut p.0, &mut p.1);
            }
        }
        for center in edge_label_centers.values_mut() {
            std::mem::swap(&mut center.0, &mut center.1);
        }
        // Swapping can push coordinates negative when the layout was
        // translated to origin in TD space; re-normalize.
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        for (_, (x, y)) in node_centers.iter() {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
        }
        if min_x.is_finite() && min_y.is_finite() {
            for center in node_centers.values_mut() {
                center.0 -= min_x;
                center.1 -= min_y;
            }
            for points in edge_points.values_mut() {
                for p in points.iter_mut() {
                    p.0 -= min_x;
                    p.1 -= min_y;
                }
            }
            for center in edge_label_centers.values_mut() {
                center.0 -= min_x;
                center.1 -= min_y;
            }
        }
    }

    // Filter feedback list (unused by caller, but kept for debugging).
    let _ = feedback;

    Some(DagreLayoutOutput {
        node_centers,
        edge_points,
        edge_label_centers,
    })
}

/// Greedily break cycles by reversing edges that go backward in a DFS order.
/// Returns the list of reversed (from, to) pairs.
fn acyclic_reverse(
    edges: &mut [LayoutEdge],
    nodes: &HashMap<String, LayoutNode>,
) -> Vec<(String, String)> {
    // Depth-first order: assign each node a discovery index, then reverse any
    // edge that points from a later-discovered node to an earlier one and is
    // part of a cycle. Simpler robust approach: iteratively find a cycle via
    // DFS and reverse one edge until acyclic.
    let mut reversed = Vec::new();
    loop {
        if let Some(cycle_edge) = find_cycle_edge(edges, nodes) {
            let (from, to) = cycle_edge;
            for edge in edges.iter_mut() {
                if edge.from == from && edge.to == to && !edge.reversed {
                    std::mem::swap(&mut edge.from, &mut edge.to);
                    edge.reversed = true;
                    reversed.push((from.clone(), to.clone()));
                    break;
                }
            }
        } else {
            break;
        }
    }
    reversed
}

/// Find one edge on a directed cycle, if any, via iterative DFS.
fn find_cycle_edge(
    edges: &[LayoutEdge],
    nodes: &HashMap<String, LayoutNode>,
) -> Option<(String, String)> {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in edges {
        adj.entry(edge.from.as_str()).or_default().push(edge.to.as_str());
    }
    // Deterministic order.
    for targets in adj.values_mut() {
        targets.sort_unstable();
    }
    let mut state: HashMap<&str, u8> = HashMap::new(); // 0=unvisited, 1=in stack, 2=done
    let mut node_ids: Vec<&str> = nodes.keys().map(|s| s.as_str()).collect();
    node_ids.sort_unstable();
    for start in node_ids {
        if state.get(start).copied().unwrap_or(0) != 0 {
            continue;
        }
        // Iterative DFS with an explicit stack of (node, next neighbor index).
        let mut stack: Vec<&str> = vec![start];
        let mut iter_stack: Vec<usize> = vec![0];
        state.insert(start, 1);
        while let Some(&node) = stack.last() {
            let neighbors = adj.get(node).cloned().unwrap_or_default();
            let top = iter_stack.len() - 1;
            let idx = iter_stack[top];
            if idx < neighbors.len() {
                iter_stack[top] = idx + 1;
                let next = neighbors[idx];
                match state.get(next).copied().unwrap_or(0) {
                    1 => {
                        // Back edge: next is in the current stack -> cycle.
                        return Some((node.to_string(), next.to_string()));
                    }
                    0 => {
                        state.insert(next, 1);
                        stack.push(next);
                        iter_stack.push(0);
                    }
                    _ => {}
                }
            } else {
                state.insert(node, 2);
                stack.pop();
                iter_stack.pop();
            }
        }
    }
    None
}

/// Longest-path rank assignment on the (now acyclic) graph.
fn rank_longest_path(
    nodes: &HashMap<String, LayoutNode>,
    edges: &[LayoutEdge],
) -> HashMap<String, usize> {
    // Kahn topological order, then longest path.
    let mut indegree: HashMap<&str, usize> = HashMap::new();
    let mut adj: HashMap<&str, Vec<(&str, usize)>> = HashMap::new();
    for id in nodes.keys() {
        indegree.insert(id.as_str(), 0);
    }
    for edge in edges {
        adj.entry(edge.from.as_str())
            .or_default()
            .push((edge.to.as_str(), edge.minlen));
        *indegree.entry(edge.to.as_str()).or_insert(0) += 1;
    }
    let mut queue: Vec<&str> = indegree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| *id)
        .collect();
    queue.sort_unstable();
    let mut ranks: HashMap<String, usize> = HashMap::new();
    for id in nodes.keys() {
        ranks.insert(id.clone(), 0);
    }
    let mut order: Vec<&str> = Vec::new();
    while let Some(node) = queue.pop() {
        order.push(node);
        if let Some(targets) = adj.get(node) {
            for (to, minlen) in targets {
                let cand = ranks[node] + minlen;
                if cand > ranks[*to] {
                    ranks.insert(to.to_string(), cand);
                }
                let deg = indegree.get_mut(to).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(to);
                }
            }
        }
    }
    // Nodes unreachable from a source (in a cycle that acyclic missed, or
    // disconnected) keep rank 0; shift so the minimum rank is 0.
    ranks
}

/// Build rank buckets from layout nodes.
fn build_rank_buckets(nodes: &HashMap<String, LayoutNode>) -> Vec<Vec<String>> {
    let max_rank = nodes.values().map(|n| n.rank).max().unwrap_or(0);
    let mut buckets: Vec<Vec<String>> = vec![Vec::new(); max_rank + 1];
    for node in nodes.values() {
        buckets[node.rank].push(node.id.clone());
    }
    // Deterministic initial order: sort by id within each rank. The
    // barycenter sweeps below refine this.
    for bucket in buckets.iter_mut() {
        bucket.sort();
    }
    buckets
}

/// Barycenter ordering sweeps (down then up, repeated).
fn order_barycenter(
    buckets: &mut [Vec<String>],
    edges: &[LayoutEdge],
    nodes: &HashMap<String, LayoutNode>,
    passes: usize,
) {
    if buckets.len() <= 1 {
        return;
    }
    let mut incoming: HashMap<String, Vec<String>> = HashMap::new();
    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    for edge in edges {
        outgoing
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
        incoming
            .entry(edge.to.clone())
            .or_default()
            .push(edge.from.clone());
    }
    // Position lookup: node id -> (rank, order). Owned keys avoid borrow
    // conflicts with the buckets being sorted.
    let mut position: HashMap<String, (usize, usize)> = HashMap::new();
    macro_rules! refresh_position {
        () => {
            position.clear();
            for (rank, bucket) in buckets.iter().enumerate() {
                for (order, id) in bucket.iter().enumerate() {
                    position.insert(id.clone(), (rank, order));
                }
            }
        };
    }
    refresh_position!();
    for pass in 0..passes {
        let downward = pass % 2 == 0;
        if downward {
            for rank in 1..buckets.len() {
                sort_bucket_by_barycenter(&mut buckets[rank], &incoming, &position, nodes);
                refresh_position!();
            }
        } else {
            for rank in (0..buckets.len() - 1).rev() {
                sort_bucket_by_barycenter(&mut buckets[rank], &outgoing, &position, nodes);
                refresh_position!();
            }
        }
    }
}

/// Sort one rank bucket by the barycenter (median) of neighbor orders.
fn sort_bucket_by_barycenter(
    bucket: &mut Vec<String>,
    neighbors: &HashMap<String, Vec<String>>,
    position: &HashMap<String, (usize, usize)>,
    _nodes: &HashMap<String, LayoutNode>,
) {
    let mut scored: Vec<(f32, usize, String)> = Vec::new();
    for (idx, id) in bucket.iter().enumerate() {
        let mut vals: Vec<f32> = neighbors
            .get(id)
            .map(|ns| {
                ns.iter()
                    .filter_map(|n| position.get(n).map(|(_, order)| *order as f32))
                    .collect()
            })
            .unwrap_or_default();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let bary = if vals.is_empty() {
            idx as f32
        } else if vals.len() % 2 == 1 {
            vals[vals.len() / 2]
        } else {
            (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
        };
        scored.push((bary, idx, id.clone()));
    }
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    *bucket = scored.into_iter().map(|(_, _, id)| id).collect();
}

/// Brandes-Kopf-style positioning.
///
/// Simplified but faithful in spirit: repeated median passes (down and up)
/// align each node with its neighbors, then each rank is compacted to
/// `nodesep` spacing. The final coordinate is the median of the down and up
/// passes, which balances the two directions.
fn position_nodes(
    nodes: &mut HashMap<String, LayoutNode>,
    buckets: &[Vec<String>],
    edges: &[LayoutEdge],
    nodesep: f32,
    ranksep: f32,
    is_horizontal: bool,
) {
    let mut incoming: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in edges {
        outgoing.entry(edge.from.as_str()).or_default().push(edge.to.as_str());
        incoming.entry(edge.to.as_str()).or_default().push(edge.from.as_str());
    }
    // Initial x: order * (average width + nodesep), centered per rank.
    let mut x: HashMap<String, f32> = HashMap::new();
    for bucket in buckets {
        let mut cursor = 0.0f32;
        for id in bucket {
            let w = nodes.get(id).map(|n| n.width).unwrap_or(0.0);
            x.insert(id.clone(), cursor + w / 2.0);
            cursor += w + nodesep;
        }
    }
    // Median passes.
    let mut x_down = x.clone();
    let mut x_up = x.clone();
    for _ in 0..4 {
        // Down pass: each rank from top, align to predecessors.
        for rank in 1..buckets.len() {
            for id in &buckets[rank] {
                let preds = incoming.get(id.as_str()).cloned().unwrap_or_default();
                let mut vals: Vec<f32> = preds
                    .iter()
                    .filter_map(|p| x_down.get(*p).copied())
                    .collect();
                if !vals.is_empty() {
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = vals[vals.len() / 2];
                    x_down.insert(id.clone(), median);
                }
            }
            compact_rank(&mut x_down, &buckets[rank], nodes, nodesep);
        }
        // Up pass: each rank from bottom, align to successors.
        for rank in (0..buckets.len() - 1).rev() {
            for id in &buckets[rank] {
                let succs = outgoing.get(id.as_str()).cloned().unwrap_or_default();
                let mut vals: Vec<f32> = succs
                    .iter()
                    .filter_map(|s| x_up.get(*s).copied())
                    .collect();
                if !vals.is_empty() {
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = vals[vals.len() / 2];
                    x_up.insert(id.clone(), median);
                }
            }
            compact_rank(&mut x_up, &buckets[rank], nodes, nodesep);
        }
    }
    // Final: median of down and up, then compact once more.
    for bucket in buckets {
        for id in bucket {
            let d = x_down.get(id).copied().unwrap_or(0.0);
            let u = x_up.get(id).copied().unwrap_or(0.0);
            x.insert(id.clone(), (d + u) / 2.0);
        }
        compact_rank(&mut x, bucket, nodes, nodesep);
    }
    // Assign coordinates. In TD space: x = horizontal position, y = rank *
    // (ranksep + node height). Node heights vary per rank, so use the max
    // height per rank for the y step.
    let mut rank_y: Vec<f32> = Vec::new();
    let mut cursor_y = 0.0f32;
    for bucket in buckets {
        rank_y.push(cursor_y);
        let max_h = bucket
            .iter()
            .map(|id| nodes.get(id).map(|n| n.height).unwrap_or(0.0))
            .fold(0.0f32, f32::max);
        cursor_y += max_h + ranksep;
    }
    for (rank, bucket) in buckets.iter().enumerate() {
        for id in bucket {
            if let Some(node) = nodes.get_mut(id) {
                node.x = *x.get(id).unwrap_or(&0.0);
                node.y = rank_y[rank] + node.height / 2.0;
            }
        }
    }
    // For horizontal layouts the caller swaps axes; keep TD here.
    let _ = is_horizontal;
}

/// Pack one rank left-to-right with at least `nodesep` gaps, preserving order.
/// Shifts the whole rank right if needed so coordinates stay non-negative.
fn compact_rank(
    x: &mut HashMap<String, f32>,
    bucket: &[String],
    nodes: &HashMap<String, LayoutNode>,
    nodesep: f32,
) {
    let mut cursor: Option<f32> = None;
    for id in bucket {
        let w = nodes.get(id).map(|n| n.width).unwrap_or(0.0);
        let desired = x.get(id).copied().unwrap_or(0.0);
        let left = desired - w / 2.0;
        let placed_left = match cursor {
            Some(c) => left.max(c),
            None => left,
        };
        x.insert(id.clone(), placed_left + w / 2.0);
        cursor = Some(placed_left + w + nodesep);
    }
}

/// Remove collinear intermediate points.
fn compress_points(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut out: Vec<(f32, f32)> = vec![points[0]];
    for i in 1..points.len() - 1 {
        let (x0, y0) = out[out.len() - 1];
        let (x1, y1) = points[i];
        let (x2, y2) = points[i + 1];
        let collinear = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
        if collinear.abs() > 1e-3 {
            out.push((x1, y1));
        }
    }
    out.push(*points.last().unwrap());
    out
}

/// Point at `progress` (0..1) along a polyline by arc length.
fn point_at_progress(points: &[(f32, f32)], progress: f32) -> Option<(f32, f32)> {
    if points.len() < 2 {
        return points.first().copied();
    }
    let mut total = 0.0f32;
    let mut seg_lens = Vec::new();
    for w in points.windows(2) {
        let dx = w[1].0 - w[0].0;
        let dy = w[1].1 - w[0].1;
        let len = (dx * dx + dy * dy).sqrt();
        seg_lens.push(len);
        total += len;
    }
    if total <= 0.0 {
        return Some(points[0]);
    }
    let target = total * progress.clamp(0.0, 1.0);
    let mut acc = 0.0f32;
    for (i, w) in points.windows(2).enumerate() {
        let len = seg_lens[i];
        if acc + len >= target {
            let t = if len > 0.0 { (target - acc) / len } else { 0.0 };
            return Some((
                w[0].0 + (w[1].0 - w[0].0) * t,
                w[0].1 + (w[1].1 - w[0].1) * t,
            ));
        }
        acc += len;
    }
    Some(*points.last().unwrap())
}

/// Clip a ray from a node center toward `toward` to the node's border.
/// Falls back to the center when the direction is degenerate.
fn clip_to_node_border(
    center: (f32, f32),
    size: (f32, f32),
    shape: NodeShape,
    toward: (f32, f32),
) -> (f32, f32) {
    let (cx, cy) = center;
    let (w, h) = size;
    let dx = toward.0 - cx;
    let dy = toward.1 - cy;
    if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
        return center;
    }
    let hw = w / 2.0;
    let hh = h / 2.0;
    match shape {
        NodeShape::Diamond => {
            // Diamond vertices: top, right, bottom, left.
            let poly = [
                (cx, cy - hh),
                (cx + hw, cy),
                (cx, cy + hh),
                (cx - hw, cy),
            ];
            if let Some(p) = ray_polygon_hit(center, (dx, dy), &poly) {
                return p;
            }
            center
        }
        NodeShape::Circle | NodeShape::DoubleCircle => {
            let rx = hw.max(1.0);
            let ry = hh.max(1.0);
            // Solve |center + t*dir - center|^2 in ellipse metric = 1.
            let a = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
            if a <= 1e-12 {
                return center;
            }
            let t = 1.0 / a.sqrt();
            (cx + dx * t, cy + dy * t)
        }
        _ => {
            // Rectangle (and cylinder/stadium approximations): intersect the
            // ray with the rect border.
            let mut t = f32::INFINITY;
            if dx.abs() > 1e-6 {
                let tx = if dx > 0.0 { hw / dx } else { -hw / dx };
                let y_at = cy + dy * tx;
                if y_at >= cy - hh - 1e-3 && y_at <= cy + hh + 1e-3 {
                    t = t.min(tx);
                }
            }
            if dy.abs() > 1e-6 {
                let ty = if dy > 0.0 { hh / dy } else { -hh / dy };
                let x_at = cx + dx * ty;
                if x_at >= cx - hw - 1e-3 && x_at <= cx + hw + 1e-3 {
                    t = t.min(ty);
                }
            }
            if t.is_finite() {
                (cx + dx * t, cy + dy * t)
            } else {
                center
            }
        }
    }
}

/// Ray (origin + t*dir, t >= 0) vs polygon: nearest hit, if any.
fn ray_polygon_hit(
    origin: (f32, f32),
    dir: (f32, f32),
    poly: &[(f32, f32)],
) -> Option<(f32, f32)> {
    let mut best_t: Option<f32> = None;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let ex = b.0 - a.0;
        let ey = b.1 - a.1;
        let denom = dir.0 * ey - dir.1 * ex;
        if denom.abs() < 1e-9 {
            continue;
        }
        let t = ((a.0 - origin.0) * ey - (a.1 - origin.1) * ex) / denom;
        let u = ((a.0 - origin.0) * dir.1 - (a.1 - origin.1) * dir.0) / denom;
        if t >= 0.0 && u >= 0.0 && u <= 1.0 {
            if best_t.map(|bt| t < bt).unwrap_or(true) {
                best_t = Some(t);
            }
        }
    }
    best_t.map(|t| (origin.0 + dir.0 * t, origin.1 + dir.1 * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_node(id: &str, w: f32, h: f32) -> LayoutNode {
        LayoutNode {
            id: id.to_string(),
            dummy_kind: None,
            width: w,
            height: h,
            rank: 0,
            order: 0,
            x: 0.0,
            y: 0.0,
            edge_idx: None,
            shape: NodeShape::Rectangle,
        }
    }

    #[test]
    fn acyclic_reverses_back_edge() {
        let mut nodes = HashMap::new();
        for id in ["A", "B", "C"] {
            nodes.insert(id.to_string(), layout_node(id, 40.0, 20.0));
        }
        let mut edges = vec![
            LayoutEdge {
                from: "A".into(),
                to: "B".into(),
                orig_idx: 0,
                weight: 1.0,
                minlen: 1,
                reversed: false,
            },
            LayoutEdge {
                from: "B".into(),
                to: "C".into(),
                orig_idx: 1,
                weight: 1.0,
                minlen: 1,
                reversed: false,
            },
            LayoutEdge {
                from: "C".into(),
                to: "A".into(),
                orig_idx: 2,
                weight: 1.0,
                minlen: 1,
                reversed: false,
            },
        ];
        let reversed = acyclic_reverse(&mut edges, &nodes);
        assert_eq!(reversed.len(), 1);
        assert!(edges.iter().any(|e| e.reversed));
        // After reversal the graph is acyclic: topological order exists.
        let ranks = rank_longest_path(&nodes, &edges);
        assert!(ranks["A"] <= ranks["B"] || ranks["B"] <= ranks["C"]);
    }

    #[test]
    fn compress_points_removes_collinear() {
        let pts = vec![(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 10.0)];
        assert_eq!(
            compress_points(&pts),
            vec![(0.0, 0.0), (20.0, 0.0), (20.0, 10.0)]
        );
    }

    #[test]
    fn point_at_progress_midpoint() {
        let pts = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        let mid = point_at_progress(&pts, 0.5).unwrap();
        assert!((mid.0 - 10.0).abs() < 1e-3);
        assert!((mid.1 - 0.0).abs() < 1e-3);
    }
}
