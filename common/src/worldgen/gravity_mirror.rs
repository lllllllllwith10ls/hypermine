use rand::{Rng, SeedableRng};
use rand_distr::{Poisson, Uniform};

use crate::{
    dodeca::Side, graph::{Graph, NodeId}, math::MVector, peer_traverser, worldgen::{hash, horosphere::NodeBoundedRegion, plane::Plane}
};

/// Whether an assortment of random flips of gravity, along with their corresponding planes, should be added to world generation. This is a temporary
/// option until large structures that fit with the theme of the world are introduced.
/// For code simplicity, this is made into a constant instead of a configuration option.
const GRAVITY_MIRRORS_ENABLED: bool = true;

/// Value to mix into the node's spice for generating gravity flips. Chosen randomly.
const GRAVITY_SEED: u64 = 6046133366614030450;

/// Represents a node's reference to a particular gravity mirror. As a general rule, for any give gravity mirror,
/// every node in the convex hull of nodes containing the gravity mirror will have a `GravityMirrorNode`
/// referencing it. The unique node in this convex hull with the smallest depth in the graph is the owner
/// of the gravity mirror, where it is originally generated.
#[derive(Copy, Clone)]
pub struct GravityMirrorNode {
    /// The node that originally created the horosphere. All parts of the horosphere will
    /// be in a node with this as an ancestor, and all GravityMirrorNodes with the same `owner` correspond
    /// to the same horosphere.
    owner: NodeId,

    /// The mirror's location relative to the node containing this `GravityMirrorNode`
    mirror: Plane,

    /// A region that bounds the `GravityMirrorNode`'s descendents. A `GravityMirrorNode` will never propagate beyond
    /// this region, and the region's bounds will be as tight as possible. Note that this region does note necessarily
    /// contain the whole horosphere because parts of the horosphere that require backtracking towards the origin
    /// are ignored.
    // Note: All public constructors generate a `GravityMirrorNode` with tight bounds, but some `GravityMirrorNode`s
    // might not have tight bounds because a `GravityMirrorNode` is used in an intermediate calculations, averaged
    // together with other `GravityMirrorNode`s before the bounds are tightened.
    region: NodeBoundedRegion,
}

impl GravityMirrorNode {
    /// Returns the `GravityMirrorNode` for the given node, either by propagating an existing parent
    /// `GravityMirrorNode` or by randomly generating a new one.
    pub fn new(graph: &Graph, node_id: NodeId) -> Option<GravityMirrorNode> {
        if !GRAVITY_MIRRORS_ENABLED {
            return None;
        }
        GravityMirrorNode::create_from_parents(graph, node_id)
            .or_else(|| GravityMirrorNode::maybe_create_fresh(graph, node_id))
    }

    /// Propagates `GravityMirrorNode` information from the given parent nodes to this child node. Returns
    /// `None` if there's no horosphere to propagate, either because none of the parent nodes have a
    /// horosphere associated with them, or because any existing horosphere is outside the range
    /// of this node.
    fn create_from_parents(graph: &Graph, node_id: NodeId) -> Option<GravityMirrorNode> {
        // Rather than selecting an arbitrary parent GravityMirrorNode, we average all of them. This
        // is important because otherwise, the propagation of floating point precision errors could
        // create a seam. This ensures that all errors average out, keeping the horosphere smooth.
        let mut mirrors_to_average_iter =
            graph
                .parents(node_id)
                .filter_map(|(parent_side, parent_id)| {
                    graph
                        .node_state(parent_id)
                        .gravity_mirror
                        .as_ref()
                        .and_then(|h| h.propagate(parent_side))
                });

        let mut mirror_node = mirrors_to_average_iter.next()?;
        let mut count = 1;
        for other in mirrors_to_average_iter {
            // Take an average of all GravityMirrorNodes in this iterator, giving each of them equal weight
            // by keeping track of a moving average with a weight that changes over time to make the
            // numbers work out the same way.
            count += 1;
            mirror_node.average_with(other, 1.0 / count as f32);
        }

        mirror_node.tighten_region_bounds();
        Some(mirror_node)
    }

    /// Create a `GravityMirrorNode` corresponding to a freshly created horosphere with the given node as its owner,
    /// if one should be created. This function is called on every node that doesn't already have a horosphere
    /// associated with it, so this function has control over how frequent the horospheres should be.
    fn maybe_create_fresh(graph: &Graph, node_id: NodeId) -> Option<GravityMirrorNode> {
        const MIRROR_DENSITY: f32 = 1.0;

        let spice = graph.hash_of(node_id) as u64;
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(hash(spice, GRAVITY_SEED));
        for _ in 0..rng.sample(Poisson::new(MIRROR_DENSITY).unwrap()) as u32 {
            // This logic is designed to create an average of "HOROSPHERE_DENSITY" horosphere candiates
            // in the region determined by `random_horosphere_pos` and then filters the resulting
            // list of candiates to only ones where the current node is the suitable owner for them.
            // Filtering instead of rejection sampling ensures a uniform distribution of horosphere
            // even though different nodes have different-sized regions for valid horospheres.

            // However, we do return early to ensure that after filtering, we only take the first
            // horosphere if there is one, since a node can have at most one horosphere.
            let mut normal: MVector<_> = 
            MVector::x() * rng.sample::<f32,_>(Uniform::new(0.0, 10.0).unwrap()) +
            MVector::y() * rng.sample::<f32,_>(Uniform::new(0.0, 10.0).unwrap()) +
            MVector::z() * rng.sample::<f32,_>(Uniform::new(0.0, 10.0).unwrap());
            normal.w = rng.sample::<f32,_>(Uniform::new(-normal.xyz().norm()/2.0, normal.xyz().norm()/2.0).unwrap());
            let mirror: Plane = normal.normalized_direction().into();
            if is_plane_valid(graph, node_id, &mirror) && graph.parents(node_id).count() > 0 &&
            graph.parents(node_id).all(|(s,p)|!(s * graph.node_state(p).surface).intersects(&mirror)){
                let mut gravity_node = GravityMirrorNode {
                    owner: node_id,
                    mirror,
                    region: NodeBoundedRegion::node_and_descendents(graph, node_id),
                };
                gravity_node.tighten_region_bounds();
                return Some(gravity_node);
            }
        }
        None
    }

    /// Updates the region associated with the `GravityMirrorNode` to have bounds that are as tight as possible.
    fn tighten_region_bounds(&mut self) {
        for side in Side::iter() {
            if !self.region.is_bounded_by(side) && self.can_tighten_region_bound(side) {
                self.region.add_bound(side);
            }
        }
    }

    /// Computes whether propagation can stop at a particular side due to no part of the horosphere
    /// being behind it. This function is used to tighten region bounds.
    fn can_tighten_region_bound(&self, side: Side) -> bool {
        !self.mirror.intersects(&side.into())
    }

    /// Returns an estimate of the `GravityMirrorNode` corresponding to the node adjacent to the current node
    /// at the given side, or `None` if the horosphere is no longer relevant after crossing the given side.
    /// The estimates given by multiple nodes may be used to produce the actual `GravityMirrorNode`.
    fn propagate(&self, side: Side) -> Option<GravityMirrorNode> {
        // Don't propagate beyond the already-computed bounds of the `GravityMirrorNode`.
        if self.region.is_bounded_by(side) {
            return None;
        }

        Some(GravityMirrorNode {
            owner: self.owner,
            mirror: side.reflection() * self.mirror,
            region: self.region.neighbor(side),
        })
    }

    /// Takes the weighted average of the coordinates of this horosphere with the coordinates of the other horosphere.
    fn average_with(&mut self, other: GravityMirrorNode, other_weight: f32) {
        if self.owner != other.owner {
            // If this panic is triggered, it may mean that two horospheres were generated that interfere
            // with each other. The logic in `should_generate` should prevent this, so this would be a sign
            // of a bug in that function's implementation.
            panic!("Tried to average two unrelated GravityMirrorNodes");
        }
        self.mirror = self.mirror.average(&other.mirror, other_weight);
        self.region = self.region.intersect(other.region);
    }

    /// Returns whether the horosphere is freshly created, instead of a
    /// reference to a horosphere created earlier on in the node graph.
    fn is_fresh(&self, node_id: NodeId) -> bool {
        self.owner == node_id
    }

    /// If `self` and `other` would propagate to the same node, to avoid interference, only one of these
    /// two horospheres can generate. This function determines whether `self` should be the one to generate.
    fn has_priority(&self, other: &GravityMirrorNode, node_id: NodeId) -> bool {
        // If both horospheres are fresh, use the owner's NodeId as an arbitrary
        // tie-breaker to decide which horosphere should win.
        !self.is_fresh(node_id) || (other.is_fresh(node_id) && self.owner < other.owner)
    }

    /// Based on other nodes in the graph, determines whether the horosphere
    /// should generate. If false, it means that another horosphere elsewhere
    /// would interfere, and generation should not proceed.
    pub fn should_generate(&self, graph: &Graph, node_id: NodeId) -> bool {
        if !self.is_fresh(node_id) {
            // The horosphere is propagated and so is already proven to exist.
            return true;
        }

        for peer in peer_traverser::expect_peer_nodes(graph, node_id) {
            let Some(peer_plane) = graph
                .partial_node_state(peer.node())
                .candidate_gravity_mirror
                .as_ref()
            else {
                continue;
            };
            if !self.has_priority(peer_plane, node_id)
                // Check that these horospheres can interfere by seeing if their regions share a node in common.
                && peer_plane.region.contains_node(peer.peer_to_shared())
                && self.region.contains_node(peer.base_to_shared())
            {
                return false;
            }
        }
        true
    }
    

    pub fn mirror(&self) -> &Plane {
        &self.mirror
    }
}

/// Returns whether the given horosphere position could represent a horosphere generated by the
/// given node. The requirement is that a horosphere must be bounded by all of the node's parent sides
/// (as otherwise, a parent node would own the horosphere), and the horosphere must not be fully
/// behind any of the other dodeca sides (as otherwise, a child node would own the horosphere). Note
/// that the horosphere does not necessarily need to intersect the dodeca to be valid.
fn is_plane_valid(graph: &Graph, node_id: NodeId, plane: &Plane) -> bool {
    (graph.parents(node_id)).all(|(s, _)| !plane.intersects(&s.into()))
}

// /// Represents a chunks's reference to a particular horosphere.
// pub struct GravityPlaneChunk {
//     /// The horosphere's location relative to the chunk containing this `HorosphereChunk`.
//     pub plane: Plane,
// }

// impl GravityPlaneChunk {
//     /// Creates a `GravityPlaneChunk` based on a `GravityMirrorNode`
//     pub fn new(plane_node: &GravityMirrorNode, vertex: Vertex) -> Self {
//         GravityPlaneChunk {
//             plane: vertex.node_to_dual() * plane_node.plane,
//         }
//     }
// }