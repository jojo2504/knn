use std::{cmp::Ordering, collections::{BinaryHeap, HashMap}};

use anyhow::Ok;
use polars::{frame::DataFrame, prelude::{AnyValue, Column, DataType, NamedFrom}, series::Series};

use crate::distance::{DistanceMetric, distance};

pub struct Neighbor {
    distance: f32,
    class: String
}

impl Neighbor {
    fn new(distance: f32, class: String) -> Self {
        Self {
            distance: distance,
            class,
        }
    }
}

impl Eq for Neighbor {
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap()
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


#[derive(Clone)]
pub struct Knn {
    pub k: u32,
    pub train_dataframe: DataFrame,
    pub metric: DistanceMetric,
    /// Gaussian kernel sigma. None = inverse-distance weighting (1/d).
    pub sigma: Option<f32>,
}

impl Knn {
    pub fn new(k: u32, train_dataframe: DataFrame) -> Self {
        Self {
            k,
            train_dataframe,
            metric: DistanceMetric::Manhattan,
            sigma: None,
        }
    }

    pub fn predict(
        &self,
        input: &Vec<AnyValue>,
        x_train: &DataFrame,
        y_train: &Column,
        class_weights: Option<&HashMap<String, f32>>,
    ) -> anyhow::Result<String> {
        let mut maxheap: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut counter: HashMap<String, f32> = HashMap::new();

        for row_train in 0..x_train.height() {
            let train_row = x_train.get_row(row_train)?.0;
            let class = y_train.get(row_train).unwrap().to_string();
            let dist = distance(input, &train_row, self.metric);
            maxheap.push(Neighbor::new(dist, class));

            if maxheap.len() > self.k as usize {
                maxheap.pop();
            }
        }

        for neighbor in maxheap.into_iter() {
            let dist_weight = if let Some(s) = self.sigma {
                (-(neighbor.distance * neighbor.distance) / (2.0 * s * s)).exp()
            } else {
                1.0 / (neighbor.distance + 1e-10)
            };
            let cls_weight = class_weights
                .and_then(|cw| cw.get(&neighbor.class).copied())
                .unwrap_or(1.0);
            *counter.entry(neighbor.class).or_insert(0.0) += dist_weight * cls_weight;
        }

        let prediction = counter
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())  // compare f32 weights
            .unwrap()
            .0
            .clone();

        Ok(prediction)
    }

    /// Compute balanced class weights: n_samples / (n_classes * class_count)
    pub fn compute_class_weights(y_train: &Column) -> HashMap<String, f32> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let n = y_train.len();
        for i in 0..n {
            let class = y_train.get(i).unwrap().to_string();
            *counts.entry(class).or_insert(0) += 1;
        }
        let n_classes = counts.len() as f32;
        counts
            .iter()
            .map(|(k, &v)| (k.clone(), n as f32 / (n_classes * v as f32)))
            .collect()
    }

    /// Add pairwise difference features (C2-C1, C3-C2, etc.) to the DataFrame.
    pub fn add_diff_features(df: &mut DataFrame) -> anyhow::Result<()> {
        let pairs: &[(&str, &str)] = &[
            ("C1", "C2"), ("C2", "C3"), ("C3", "C4"),
            ("C5", "C6"), ("C6", "C7"),
        ];
        for &(a, b) in pairs {
            let ca: Vec<f64> = df.column(a)?.f64()?.into_iter().map(|v| v.unwrap()).collect();
            let cb: Vec<f64> = df.column(b)?.f64()?.into_iter().map(|v| v.unwrap()).collect();
            let diff: Vec<f64> = ca.iter().zip(cb.iter()).map(|(x, y)| x - y).collect();
            let col_name = format!("diff_{}_{}", a, b);
            df.with_column(Series::new(col_name.as_str().into(), diff).into())?;
        }
        Ok(())
    }

    /// Add ALL 21 pairwise difference features between C1..C7.
    pub fn add_all_diff_features(df: &mut DataFrame) -> anyhow::Result<()> {
        let cols = ["C1","C2","C3","C4","C5","C6","C7"];
        for i in 0..cols.len() {
            for j in (i+1)..cols.len() {
                let ca: Vec<f64> = df.column(cols[i])?.f64()?.into_iter().map(|v| v.unwrap()).collect();
                let cb: Vec<f64> = df.column(cols[j])?.f64()?.into_iter().map(|v| v.unwrap()).collect();
                let diff: Vec<f64> = ca.iter().zip(cb.iter()).map(|(x, y)| x - y).collect();
                let col_name = format!("diff_{}_{}", cols[i], cols[j]);
                df.with_column(Series::new(col_name.as_str().into(), diff).into())?;
            }
        }
        Ok(())
    }

    /// Add ratio features C2/C1, C3/C1, C3/C2, C4/C1 — key discriminators between class 0 and class 3.
    pub fn add_ratio_features(df: &mut DataFrame) -> anyhow::Result<()> {
        let pairs: &[(&str, &str, &str)] = &[
            ("C2", "C1", "ratio_C2_C1"),
            ("C3", "C1", "ratio_C3_C1"),
            ("C3", "C2", "ratio_C3_C2"),
            ("C4", "C1", "ratio_C4_C1"),
            ("C3", "C4", "ratio_C3_C4"),
        ];
        for &(num, den, name) in pairs {
            let ca: Vec<f64> = df.column(num)?.f64()?.into_iter().map(|v| v.unwrap()).collect();
            let cb: Vec<f64> = df.column(den)?.f64()?.into_iter().map(|v| v.unwrap()).collect();
            let ratio: Vec<f64> = ca.iter().zip(cb.iter()).map(|(x, y)| x / (y + 1e-10)).collect();
            df.with_column(Series::new(name.into(), ratio).into())?;
        }
        Ok(())
    }

    /// Scale C5, C6, C7 by `weight` after they have been added (call after transform).
    pub fn amplify_features(df: &mut DataFrame, cols: &[&str], weight: f64) -> anyhow::Result<()> {
        for &col in cols {
            let vals: Vec<f64> = df.column(col)?.f64()?.into_iter()
                .map(|v| v.unwrap() * weight).collect();
            df.replace(col, Series::new(col.into(), vals).into())?;
        }
        Ok(())
    }

    // Scale all features within the given dataframe.
    pub fn transform(df: &mut DataFrame, train_stats: Option<Vec<(f64, f64)>>) -> anyhow::Result<Option<Vec<(f64, f64)>>> {
        // Auto-detect all Float64 columns in DataFrame column order
        let feature_cols: Vec<String> = df
            .schema()
            .iter()
            .filter(|(_, dtype)| **dtype == DataType::Float64)
            .map(|(name, _)| name.to_string())
            .collect();
        let mut stats = Vec::new();

        for (index, col_name) in feature_cols.iter().enumerate() {
            let (mean, std) = if let Some(ref s) = train_stats {
                (s[index].0, s[index].1)
            } else {
                let c = df.column(col_name.as_str())?;
                let mean = c.mean_reduce()?;
                let mean = mean.value().try_extract::<f64>()?;
                let std  = c.std_reduce(1)?;
                let std  = std.value().try_extract::<f64>()?;
                stats.push((mean, std));
                (mean, std)
            };

            let values: Vec<f64> = df
                .column(col_name.as_str())?
                .f64()?
                .into_iter()
                .map(|v: Option<f64>| (v.unwrap() - mean) / std)
                .collect();

            let new_col = Series::new(col_name.as_str().into(), values);
            df.replace(col_name.as_str(), new_col.into())?;
        }

        if stats.is_empty() {
            return Ok(None);
        }
        Ok(Some(stats))
    }
}