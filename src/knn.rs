use std::{cmp::Ordering, collections::{BinaryHeap, HashMap}};

use anyhow::Ok;
use polars::{frame::DataFrame, prelude::{AnyValue, Column, NamedFrom}, series::Series};

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
    pub train_dataframe: DataFrame
}

impl Knn {
    pub fn new(k: u32, train_dataframe: DataFrame) -> Self {
        Self {
            k,            
            train_dataframe 
        }
    }

    pub fn predict(&self, input: &Vec<AnyValue>, x_train: &DataFrame, y_train: &Column) -> anyhow::Result<String> {
        let mut maxheap: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut counter: HashMap<String, f32> = HashMap::new();

        for row_train in 0..x_train.height() {
            let train_row = x_train.get_row(row_train)?.0;
            let class = y_train.get(row_train).unwrap().to_string();
            let distance = distance(&input, &train_row, DistanceMetric::Manhattan);
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

