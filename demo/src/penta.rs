use std::f64::consts::TAU;

use enum_map::{enum_map, Enum, EnumMap};
use lazy_static::lazy_static;

use crate::math::HyperboloidVector;

lazy_static! {
    static ref NEIGHBORHOOD: EnumMap<Side, EnumMap<Side, bool>> = enum_map! {
        Side::A => enum_map! { Side::E | Side::B => true, _ => false },
        Side::B => enum_map! { Side::A | Side::C => true, _ => false },
        Side::C => enum_map! { Side::B | Side::D => true, _ => false },
        Side::D => enum_map! { Side::C | Side::E => true, _ => false },
        Side::E => enum_map! { Side::D | Side::A => true, _ => false },
    };
    static ref PENTA: Penta = Penta::compute();
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Enum)]
pub enum Side {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
}

impl Side {
    pub fn iter() -> impl ExactSizeIterator<Item = Self> {
        use Side::*;
        [A, B, C, D, E].iter().cloned()
    }

    #[inline]
    pub fn is_neighbor(self, side: Side) -> bool {
        NEIGHBORHOOD[self][side]
    }

    #[inline]
    pub fn reflection(self) -> &'static na::Matrix3<f64> {
        &PENTA.reflections[self]
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Enum)]
pub enum Vertex {
    AB = 0,
    BC = 1,
    CD = 2,
    DE = 3,
    EA = 4,
}

impl Vertex {
    pub fn iter() -> impl ExactSizeIterator<Item = Self> {
        use Vertex::*;
        [AB, BC, CD, DE, EA].iter().cloned()
    }

    #[inline]
    pub fn voxel_to_hyperboloid(self) -> &'static na::Matrix3<f64> {
        &PENTA.voxel_to_hyperboloid[self]
    }

    #[inline]
    pub fn hyperboloid_to_voxel(self) -> &'static na::Matrix3<f64> {
        &PENTA.hyperboloid_to_voxel[self]
    }
}

struct Penta {
    vertex_sides: EnumMap<Vertex, (Side, Side)>,
    normals: EnumMap<Side, na::Vector3<f64>>,
    reflections: EnumMap<Side, na::Matrix3<f64>>,
    vertices: EnumMap<Vertex, na::Vector3<f64>>,
    voxel_to_hyperboloid: EnumMap<Vertex, na::Matrix3<f64>>,
    hyperboloid_to_voxel: EnumMap<Vertex, na::Matrix3<f64>>,
}

impl Penta {
    fn compute() -> Self {
        // Order 4 pentagonal tiling
        // Note: Despite being constants, they are not really configurable, as the rest of the code
        // depends on them being set to their current values, NUM_SIDES = 5 and ORDER = 4
        const NUM_SIDES: usize = 5;
        const ORDER: usize = 4;

        let side_angle = TAU / NUM_SIDES as f64;
        let order_angle = TAU / ORDER as f64;

        let cos_side_angle = side_angle.cos();
        let cos_order_angle = order_angle.cos();

        let reflection_r = ((1.0 + cos_order_angle) / (1.0 - cos_side_angle)).sqrt();
        let reflection_z = ((cos_side_angle + cos_order_angle) / (1.0 - cos_side_angle)).sqrt();

        let vertex_sides: EnumMap<Vertex, (Side, Side)> = enum_map! {
            Vertex::AB => (Side::A, Side::B),
            Vertex::BC => (Side::B, Side::C),
            Vertex::CD => (Side::C, Side::D),
            Vertex::DE => (Side::D, Side::E),
            Vertex::EA => (Side::E, Side::A),
        };

        let mut normals: EnumMap<Side, na::Vector3<f64>> = EnumMap::default();
        let mut vertices: EnumMap<Vertex, na::Vector3<f64>> = EnumMap::default();
        let mut voxel_to_hyperboloid: EnumMap<Vertex, na::Matrix3<f64>> = EnumMap::default();

        for (side, reflection) in normals.iter_mut() {
            let theta = side_angle * (side as usize) as f64;
            *reflection = na::Vector3::new(
                reflection_r * theta.cos(),
                reflection_r * theta.sin(),
                reflection_z,
            );
        }

        for (vertex, vertex_pos) in vertices.iter_mut() {
            *vertex_pos = normals[vertex_sides[vertex].0]
                .normal(&normals[vertex_sides[vertex].1]);
            *vertex_pos /= vertex_pos.sqr().sqrt();
        }

        for (vertex, mat) in voxel_to_hyperboloid.iter_mut() {
            let reflector0 = &normals[vertex_sides[vertex].0];
            let reflector1 = &normals[vertex_sides[vertex].1];
            let origin = na::Vector3::new(0.0, 0.0, 1.0);
            *mat = na::Matrix3::from_columns(&[
                -reflector0 * reflector0.z,
                -reflector1 * reflector1.z,
                origin + reflector0 * reflector0.z + reflector1 * reflector1.z,
            ]);
        }

        Penta {
            vertex_sides,
            normals,
            reflections: normals.map(|_, v| v.reflection()),
            vertices,
            voxel_to_hyperboloid,
            hyperboloid_to_voxel: voxel_to_hyperboloid.map(|_, m| m.try_inverse().unwrap()),
        }
    }
}
