use std::{cmp::Ordering, collections::{BinaryHeap, HashMap}};

use anyhow::Ok;
use polars::{frame::DataFrame, prelude::{AnyValue, Column, DataType, NamedFrom}, series::Series};

use crate::distance::{DistanceMetric, distance};

// Feature offsets discovered from data analysis.
// Each raw feature = integer_part + fixed_offset.
// Subtracting these before scaling exposes the true integer structure.
const OFFSETS: [f64; 7] = [
    412.8875226973147,  // C1
    530.7100921881075,  // C2
    657.7109720097937,  // C3
    308.78135616697125, // C4
    184.73614303700987, // C5
    78.9679071935397,   // C6
    382.42649089380166, // C7
];

// Scaled thresholds for the ambiguous B=(0,0,0) region:
// C1_int<=20, C2_int<=40, C3_int<=87 — the only zone where class 0 and
// class 3 overlap. In this region k=1 outperforms k=3 because the class-3
// samples are sparse (21 vs 805 class-0) and k=3 votes are dominated by
// the majority class. k=1 correctly follows the nearest neighbor in C5-C7 space.
const AMBIGUOUS_C1_THRESH: f32 = -0.11807846;
const AMBIGUOUS_C2_THRESH: f32 =  0.18362583;
const AMBIGUOUS_C3_THRESH: f32 =  0.86402361;

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
    pub feature_importances: Option<Vec<f64>>
}

impl Knn {
    pub fn new(k: u32, train_dataframe: DataFrame, compute_importances: bool) -> Self {
        let mut knn = Self {
            k,            
            train_dataframe,
            feature_importances: None
        };

        if compute_importances {
            let result: Vec<(String, f64)> = knn.feature_importance().unwrap();
            let mut features_weight: Vec<f64> = result
                .iter()
                .map(|(_, b)| {
                    *b
                })
                .collect();
            
            let total: f64 = features_weight.iter().sum();
            features_weight
                .iter_mut()
                .for_each(|x| *x /= total);
            
            knn.feature_importances = Some(features_weight);
        }

        knn
    }

    pub fn predict(&self, input: &Vec<AnyValue>, x_train: &DataFrame, y_train: &Column) -> anyhow::Result<String> {
        // Use k=1 in the ambiguous B=(0,0,0) region where class 0 and class 3
        // overlap and k=3 is dominated by the majority class-0 density.
        let effective_k = {
            let c1 = input[0].extract::<f32>().unwrap_or(f32::MAX);
            let c2 = input[1].extract::<f32>().unwrap_or(f32::MAX);
            let c3 = input[2].extract::<f32>().unwrap_or(f32::MAX);
            if c1 <= AMBIGUOUS_C1_THRESH && c2 <= AMBIGUOUS_C2_THRESH && c3 <= AMBIGUOUS_C3_THRESH {
                1
            } else {
                self.k as usize
            }
        };

        let mut maxheap: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut counter: HashMap<String, f32> = HashMap::new();

        for row_train in 0..x_train.height() {
            let train_row = x_train.get_row(row_train)?.0;
            let class = y_train.get(row_train).unwrap().to_string();
            let distance = distance(&input, &train_row, DistanceMetric::Manhattan, &self.feature_importances);
            maxheap.push(Neighbor::new(distance, class));

            if maxheap.len() > effective_k {
                maxheap.pop();
            }
        }

        for neighbor in maxheap.into_iter() {  // extract elements
            let weight = 1.0 / (neighbor.distance + 1e-10);
            *counter.entry(neighbor.class).or_insert(0.0) += weight;
        }

        let prediction = counter
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())  // compare f32 weights
            .unwrap()
            .0
            .clone();

        Ok(prediction)
    }

    /// Compute Fisher's Discriminant Ratio (FDR) for every numeric feature column
    /// against a label column.
    ///
    /// FDR = Σ_c [ n_c * (μ_c − μ_global)² ] / Σ_c [ n_c * σ_c² ]
    ///
    /// A higher FDR means the feature separates the classes better.
    /// Also prints per-class means so you can spot patterns like "C1 > 500 → class 2".
    pub fn feature_importance(&self) -> anyhow::Result<Vec<(String, f64)>> {
        let y = self.train_dataframe.column("Label")?;

        // Collect unique classes and their row indices
        let n = self.train_dataframe.height();
        let mut class_rows: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let cls = y.get(i)?.to_string();
            class_rows.entry(cls).or_default().push(i);
        }

        let feature_cols: Vec<String> = self.train_dataframe
            .schema()
            .iter()
            .filter(|(name, dtype)| **dtype == DataType::Float64 && name.as_str() != "Label")
            .map(|(name, _)| name.to_string())
            .collect();

        let mut results: Vec<(String, f64)> = Vec::new();

        for col_name in &feature_cols {
            let col: Vec<f64> = self.train_dataframe.column(col_name)?.f64()?.into_iter()
                .map(|v| v.unwrap_or(0.0)).collect();

            let global_mean = col.iter().sum::<f64>() / n as f64;

            let mut between = 0.0f64;
            let mut within  = 0.0f64;

            let mut class_means: Vec<(String, f64)> = Vec::new();

            for (cls, rows) in &class_rows {
                let nc = rows.len() as f64;
                let mean_c = rows.iter().map(|&i| col[i]).sum::<f64>() / nc;
                let var_c  = rows.iter().map(|&i| (col[i] - mean_c).powi(2)).sum::<f64>() / nc;
                between += nc * (mean_c - global_mean).powi(2);
                within  += nc * var_c;
                class_means.push((cls.clone(), mean_c));
            }

            let fdr = if within.abs() < 1e-12 { f64::INFINITY } else { between / within };
            class_means.sort_by(|a, b| a.0.cmp(&b.0));
            results.push((col_name.clone(), fdr));
        }

        Ok(results)
    }

    // Scale all features within the given dataframe.
    // Offsets are subtracted first to expose the integer structure of the data,
    // then standard mean/std normalisation is applied as before.
    pub fn transform(df: &mut DataFrame, train_stats: Option<Vec<(f64, f64)>>) -> anyhow::Result<Option<Vec<(f64, f64)>>> {
        let feature_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];
        let mut stats = Vec::new();

        for (index, col_name) in feature_cols.iter().enumerate() {
            let offset = OFFSETS[index];

            let (mean, std, c) = if let Some(stats) = &train_stats {
                let mean = stats[index].0;
                let std = stats[index].1;
                let c = df.column(col_name)?;
                (mean, std, c)
            } else {
                // Subtract offset first, then compute mean/std on the integer parts
                let c = df.column(col_name)?;
                let int_parts: Vec<f64> = c.f64()?.into_iter()
                    .map(|v| v.unwrap() - offset)
                    .collect();
                let mean = int_parts.iter().sum::<f64>() / int_parts.len() as f64;
                let std = (int_parts.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (int_parts.len() - 1) as f64)
                    .sqrt();
                stats.push((mean, std));
                (mean, std, c)
            };

            let c = c.f64()?;
            let values: Vec<f64> = c
                .into_iter()
                .map(|v| v.unwrap())
                .map(|v| ((v - offset) - mean) / std)  // subtract offset, then standardize
                .collect();

            let new_col = Series::new((*col_name).into(), values);
            df.replace(col_name, new_col.into())?;
        }

        if stats.is_empty() {
            return Ok(None);
        }
        Ok(Some(stats))
    }
}