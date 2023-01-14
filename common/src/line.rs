use std::ops::{Mul, Neg};

use crate::{
    dodeca::{Side, Vertex},
    math::{lorentz_normalize, mip, translate_along, origin},
    Plane,
};

/// A hyperbolic line
#[derive(Debug, Copy, Clone)]
pub struct Line<N: na::RealField> {
    vector: na::Vector4<N>,
}

impl From<Side> for Line<f64> {
    /// A surface overlapping with a particular dodecahedron side
    fn from(side: Side) -> Self {
        Self {
            vector: *side.normal(),
        }
    }
}

impl<N: na::RealField + Copy> From<na::Unit<na::Vector3<N>>> for Line<N> {
    /// A line passing through the origin
    fn from(x: na::Unit<na::Vector3<N>>) -> Self {
        Self {
            vector: x.into_inner().push(na::zero()),
        }
    }
}

impl<N: na::RealField + Copy> Neg for Line<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            vector: -self.vector,
        }
    }
}


impl<N: na::RealField + Copy> Line<N> {
    /// Hyperbolic vector identifying the line
    pub fn vector(&self) -> &na::Vector4<N> {
        &self.vector
    }

    /// Shortest distance between the line and a point
    pub fn distance_to(&self, point: &na::Vector4<N>) -> N {
        let vec = na::Vector3::new(self.vector[0], self.vector[1], self.vector[2]).normalize();
        let plane = Plane::from(nalgebra::Unit::new_normalize(vec));
        let mat = translate_along(&nalgebra::Unit::new_normalize(vec),-plane.distance_to(point));
        let mip_value = mip(&(mat*point),&origin()).abs().acosh();
        // Workaround for bug fixed in rust PR #72486
        mip_value
    }
}

impl Line<f64> {
    /// Like `distance_to`, but using chunk coordinates for a chunk in the same node space
    pub fn distance_to_chunk(&self, chunk: Vertex, coord: &na::Vector3<f64>) -> f64 {
        let pos = lorentz_normalize(&(chunk.chunk_to_node() * coord.push(1.0)));
        self.distance_to(&pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{origin, translate_along};
    use approx::*;

    #[test]
    fn distance_sanity() {
        for &axis in &[
            na::Vector3::x_axis(),
            na::Vector3::y_axis(),
            na::Vector3::z_axis(),
        ] {
            for &axis2 in &[
                na::Vector3::x_axis(),
                na::Vector3::y_axis(),
                na::Vector3::z_axis(),
            ] {
                for &distance in &[-1.5f32, 0.0f32, 1.5f32] {
                    if axis != axis2 {
                        let line = Line::from(axis);
                        assert_abs_diff_eq!(
                            line.distance_to(&(translate_along(&axis2, distance) * origin())),
                            distance.abs()
                        );
                    }
                }
            }
        }
    }
}