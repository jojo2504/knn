use polars::prelude::AnyValue;

pub fn euclidian_distance(x1: &Vec<AnyValue>, x2: &Vec<AnyValue>) -> f32 {
    let _x1 = &x1[..];
    let _x2 = &x2[..];

    _x1.iter()
        .zip(_x2.iter())
        .map(|(a, b)| {
            let diff = a.extract::<f32>().unwrap() - b.extract::<f32>().unwrap();
            diff.powi(2)
        })
        .sum::<f32>()
        .sqrt()
}

pub fn manhattan_distance(x1: &Vec<AnyValue>, x2: &Vec<AnyValue>) -> f32 {
    let _x1 = &x1[..];
    let _x2 = &x2[..];

    _x1.iter()
        .zip(_x2.iter())
        .map(|(a, b)| {
            (a.extract::<f32>().unwrap() - b.extract::<f32>().unwrap()).abs()
        })
        .sum::<f32>()
}

pub fn minkowski_distance(x1: &Vec<AnyValue>, x2: &Vec<AnyValue>, p: i32) -> f32 {
    let _x1 = &x1[..];
    let _x2 = &x2[..];

    _x1.iter()
        .zip(_x2.iter())
        .map(|(a, b)| {
            (a.extract::<f32>().unwrap() - b.extract::<f32>().unwrap()).abs().powi(p)
        })
        .sum::<f32>()
        .powf(1.0 / p as f32)
}

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski(i32),
}

pub fn distance(x1: &Vec<AnyValue>, x2: &Vec<AnyValue>, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean    => euclidian_distance(x1, x2),
        DistanceMetric::Manhattan    => manhattan_distance(x1, x2),
        DistanceMetric::Minkowski(p) => minkowski_distance(x1, x2, p),
    }
}