use std::ops::Neg;

use crate::{
    dodeca::{Side, Vertex},
    math::MVector,
};

/// A hyperbolic line
#[derive(Debug, Copy, Clone)]
pub struct Line<N: na::RealField> {
    vector: MVector<N>,
    point: MVector<N>,
}

impl From<Side> for Line<f64> {
    /// A surface overlapping with a particular dodecahedron side
    fn from(side: Side) -> Self {
        Self {
            vector: *side.normal_f64(),
            point: MVector::origin(),
        }
    }
}

impl<N: na::RealField + Copy> From<na::Unit<na::Vector3<N>>> for Line<N> {
    /// A line passing through the origin
    fn from(x: na::Unit<na::Vector3<N>>) -> Self {
        Self {
            vector: MVector::from(x),
            point: MVector::origin(),
        }
    }
}

impl<N: na::RealField + Copy> Neg for Line<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            vector: -self.vector,
            point: self.point,
        }
    }
}


impl<N: na::RealField + Copy> Line<N> {
    /// Hyperbolic vector identifying the line
    pub fn vector(&self) -> &MVector<N> {
        &self.vector
    }
    
    /// Hyperbolic point identifying the line
    pub fn point(&self) -> &MVector<N> {
        &self.point
    }

    /// Shortest distance between the line and a point
    pub fn distance_to(&self, point: &MVector<N>) -> N {
        (point.mip(&self.point).powi(2) - point.mip(&self.vector).powi(2))
            .sqrt()
            .acosh()
    }
    pub fn from_points(point1: &MVector<N>, point2: &MVector<N>) -> Self {
        let point1 = point1.lorentz_normalize();
        let point_diff = *point2 - point1;
        let a = &(point_diff + point1 * point1.mip(&point_diff)).lorentz_normalize();
        Self {
            vector: *a,
            point: point1,
        }
    }
    pub fn new(p: &MVector<N>, d: na::Unit<na::Vector3<N>>) -> Self {
        Self {
            vector: MVector::from(d),
            point: *p,
        }
    }
}

impl Line<f64> {
    /// Like `distance_to`, but using chunk coordinates for a chunk in the same node space
    pub fn distance_to_chunk(&self, chunk: Vertex, coord: &na::Vector3<f64>) -> f64 {
        let pos = (MVector::from(chunk.chunk_to_node_f64() * coord.push(1.0))).lorentz_normalize();
        self.distance_to(&pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::translate_along;
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
                            line.distance_to(&(translate_along(&(*axis2*distance)) * MVector::origin())),
                            distance.abs()
                        );
                    }
                }
            }
        }
    }
}