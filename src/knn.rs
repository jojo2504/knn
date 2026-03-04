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
        let mut maxheap: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut counter: HashMap<String, f32> = HashMap::new();

        for row_train in 0..x_train.height() {
            let train_row = x_train.get_row(row_train)?.0;
            let class = y_train.get(row_train).unwrap().to_string();
            let distance = distance(&input, &train_row, DistanceMetric::Manhattan, &self.feature_importances);
            maxheap.push(Neighbor::new(distance, class));

            if maxheap.len() > self.k as usize {
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
    pub fn transform(df: &mut DataFrame, train_stats: Option<Vec<(f64, f64)>>) -> anyhow::Result<Option<Vec<(f64, f64)>>> {
        let feature_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];
        let mut stats = Vec::new();

        for (index, col_name) in feature_cols.iter().enumerate() {
            let (mean, std, c) = if let Some(stats) = &train_stats {
            let mean = stats[index].0;
            let std = stats[index].1;
            let c = df.column(col_name)?; // get column
            (mean, std, c)
        } else {
            let c = df.column(col_name)?;
            let mean = c.mean_reduce()?;
            let mean = mean.value().try_extract::<f64>()?;
            let std = c.std_reduce(1)?;
            let std = std.value().try_extract::<f64>()?;
            stats.push((mean, std));
            (mean, std, c)
        };

        // now `c`, `mean`, `std` are all available here
        let c = c.f64()?;
        let values: Vec<f64> = c
            .into_iter()
            .map(|v| v.unwrap())
            .map(|v| (v - mean) / std)
            .collect();
        
            let new_col = Series::new((*col_name).into(), values);
            df.replace(col_name, new_col.into())?;
        }

        // println!("{}", self.train_dataframe.head(Some(5)));
        if stats.is_empty() {
            return Ok(None);
        }
        Ok(Some(stats))
    }
}

