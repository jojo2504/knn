pub mod distance;
pub mod knn;

use std::{env, fs::File, io::{BufWriter, Write}, path::Path};

use anyhow::Ok;
use knn::Knn;
use polars::{io::SerReader, prelude::CsvReadOptions};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let assets_directory = Path::new(&args[1]);
    
    let train = CsvReadOptions::default()
    .with_has_header(true)
    .try_into_reader_with_file_path(Some(
        assets_directory.join("train.csv"),
    ))?
    .finish()?
    .drop_nulls::<String>(None)?;

    let mut input = CsvReadOptions::default()
    .with_has_header(true)
    .try_into_reader_with_file_path(Some(
        assets_directory.join("test.csv"),
    ))?
    .finish()?
    .drop_nulls::<String>(None)?;

    let knn = Knn::new(3, train, true);

    let mut x_train = knn.train_dataframe
        .select(knn.train_dataframe.get_column_names()[1..8].to_vec())?;
    let y_train = knn.train_dataframe.column("Label")?;
    
    let stats = Knn::transform(&mut x_train, None)?;
    Knn::transform(&mut input, stats)?;

    let results: Vec<String> = (0..input.height())
    .into_par_iter()
    .map(|row_test| {
        let row = input.get_row(row_test).unwrap(); // Get row
        let mut input_row = row.0;
        
        let id = input_row[0].clone();
        input_row.remove(0);

        let prediction = knn.predict(&input_row, &x_train, y_train).unwrap();
        format!("{},{}", id, prediction)
    })
    .collect();

    let file = File::create(assets_directory.join("submission.csv"))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "Id,Label")?;

    writer.write_all(results.join("\n").as_bytes())?;
    writer.flush()?;

    Ok(())
}







#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};
    use polars::{
        io::SerReader,
        prelude::{AnyValue, CsvReadOptions, NamedFrom},
        series::Series,
    };
    use rand::{rng, seq::SliceRandom};
    use colored::Colorize;

    use crate::knn::Knn;

    /// Returns the assets directory from the `ASSETS_DIR` environment variable.
    /// Usage: `ASSETS_DIR=/path/to/assets cargo test -- --nocapture`
    fn assets_dir() -> PathBuf {
        PathBuf::from(
            env::var("ASSETS_DIR")
                .expect("Set the ASSETS_DIR environment variable to your assets directory path\nExample: ASSETS_DIR=/path/to/assets cargo test find_best_k -- --nocapture\n"),
        )
    }

    #[test]
    pub fn test_knn_with_train_df() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(
                assets_dir().join("train.csv"),
            ))?
            .finish()?
            .drop_nulls::<String>(None)?;

        let mut knn = Knn::new(3, train, true);

        // Shuffle the training df to not over train it
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        knn.train_dataframe = knn.train_dataframe.take(indices_series.u32()?)?;

        // train based on 20% of the df
        let x = knn
            .train_dataframe
            .select(knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = knn
            .train_dataframe
            .column(knn.train_dataframe.get_column_names()[8])?;

        let n: usize = knn.train_dataframe.height();
        let train_size = (n as f64 * 0.8) as usize;

        let mut x_train = x.slice(0, train_size);
        let y_train = y.slice(0, train_size);
        
        let mut x_test = x.slice(train_size as i64, n - train_size);
        let y_test = y.slice(train_size as i64, n - train_size);
        
        let stats = Knn::transform(&mut x_train, None)?;
        Knn::transform(&mut x_test, stats)?;

        let mut correct = 0;
        let mut total = 0;
        for row_test in 0..x_test.height() {
            let inputs: Vec<AnyValue> = x_test.get_row(row_test)?.0;
            let prediction = knn.predict(&inputs, &x_train, &y_train)?;

            let real = y_test.get(row_test)?.to_string();
            if prediction == real {
                correct += 1;
                println!(
                    "{}",
                    &format!("prediction : {}, Real: {}", prediction, y_test.get(row_test)?).green()
                );
            } else {
                println!(
                    "{}",
                    &format!("prediction : {}, Real: {}", prediction, y_test.get(row_test)?).red()
                );
            }
            total += 1;

            println!(
                "Accuracy: {}/{} = {:.2}%",
                correct,
                total,
                (correct as f32 / total as f32) * 100.0
            );
        }
        Ok(())
    }

    #[test]
    pub fn find_best_k() -> anyhow::Result<()> {
        let base_train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(
                assets_dir().join("train.csv"),
            ))?
            .finish()?
            .drop_nulls::<String>(None)?;
        
        let mut base_knn = Knn::new(0, base_train, true.clone());
        
        // Shuffle the training df to not over train it
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..base_knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        base_knn.train_dataframe = base_knn.train_dataframe.take(indices_series.u32()?)?;

        // train based on 20% of the df
        let x = base_knn
            .train_dataframe
            .select(base_knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = base_knn
            .train_dataframe
            .column(base_knn.train_dataframe.get_column_names()[8])?;

        let n: usize = base_knn.train_dataframe.height();
        let train_size = (n as f64 * 0.8) as usize;

        let mut x_train = x.slice(0, train_size);
        let y_train = y.slice(0, train_size);
        
        let mut x_test = x.slice(train_size as i64, n - train_size);
        let y_test = y.slice(train_size as i64, n - train_size);

        let stats = Knn::transform(&mut x_train, None)?;
        Knn::transform(&mut x_test, stats)?;

        let mut best_accuracy = 0.0;
        let mut best_k = 0;
        for k in 2..15 {
            let mut knn = base_knn.clone();
            knn.k = k;
            
            let mut correct = 0;
            let mut total = 0;
            for row_test in 0..x_test.height() {
                let inputs: Vec<AnyValue> = x_test.get_row(row_test)?.0;
                let prediction = knn.predict(&inputs, &x_train, &y_train)?;
    
                let real = y_test.get(row_test)?.to_string();
                if prediction == real {
                    correct += 1;
                } 
                total += 1;
    
            }

            let current_accuracy = (correct as f32 / total as f32) * 100.0;

            if current_accuracy > best_accuracy {
                best_k = k;
                best_accuracy = current_accuracy;
            } 

            println!(
                "Accuracy: {}/{} = {:.2}% with k={}",
                correct,
                total,
                current_accuracy,
                k
            );
        }
        println!("best k={} with accuracy={}", best_k, best_accuracy);
        Ok(())
    }

    #[test]
    pub fn find_best_k_fold() -> anyhow::Result<()> {
        let base_train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(
                assets_dir().join("train.csv"),
            ))?
            .finish()?
            .drop_nulls::<String>(None)?;

        let mut base_knn = Knn::new(0, base_train, true.clone());

        // Shuffle once
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..base_knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        base_knn.train_dataframe = base_knn.train_dataframe.take(indices_series.u32()?)?;

        let x = base_knn
            .train_dataframe
            .select(base_knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = base_knn
            .train_dataframe
            .column(base_knn.train_dataframe.get_column_names()[8])?
            .clone();

        let n = base_knn.train_dataframe.height();
        let num_folds = 5;
        let fold_size = n / num_folds;

        let mut best_accuracy = 0.0f32;
        let mut best_k = 0;

        for k in 2..6 {
            let mut fold_accuracies: Vec<f32> = Vec::new();

            for fold in 0..num_folds {
                let test_start  = fold * fold_size;
                let test_end    = test_start + fold_size;

                // Test fold
                let x_test = x.slice(test_start as i64, fold_size);
                let y_test = y.slice(test_start as i64, fold_size);

                // Train = everything before + everything after test fold
                let x_train_a = x.slice(0, test_start);
                let x_train_b = x.slice(test_end as i64, n - test_end);
                let mut y_train_a = y.slice(0, test_start);
                let y_train_b = y.slice(test_end as i64, n - test_end);

                let mut x_train = x_train_a.vstack(&x_train_b)?;
                let y_train = y_train_a.extend(&y_train_b)?;

                // Scale — fit on train, apply to test
                let stats = Knn::transform(&mut x_train, None)?;
                let mut x_test = x_test; // make mutable
                Knn::transform(&mut x_test, stats)?;

                // Predict
                let mut knn = base_knn.clone();
                knn.k = k;

                let mut correct = 0;
                for row_test in 0..x_test.height() {
                    let inputs: Vec<AnyValue> = x_test.get_row(row_test)?.0;
                    let prediction = knn.predict(&inputs, &x_train, &y_train)?;
                    let real = y_test.get(row_test)?.to_string();
                    if prediction == real {
                        correct += 1;
                    }
                }

                let fold_accuracy = correct as f32 / fold_size as f32 * 100.0;
                println!("  fold={} k={} accuracy={:.2}%", fold + 1, k, fold_accuracy);
                fold_accuracies.push(fold_accuracy);
            }

            // Average accuracy across all folds
            let mean_accuracy = fold_accuracies.iter().sum::<f32>() / num_folds as f32;
            println!("k={} mean_accuracy={:.2}%\n", k, mean_accuracy);

            if mean_accuracy > best_accuracy {
                best_accuracy = mean_accuracy;
                best_k = k;
            }
        }

        println!("best k={} with mean accuracy={:.2}%", best_k, best_accuracy);
        Ok(())
    }

    /// 10-fold cross-validation — prints each prediction green/red, then fold summary.
    /// Usage: ASSETS_DIR=/path/to/assets cargo test cross_validate_10_fold --release -- --nocapture
    #[test]
    pub fn cross_validate_10_fold() -> anyhow::Result<()> {
        let base_train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(
                assets_dir().join("train.csv"),
            ))?
            .finish()?
            .drop_nulls::<String>(None)?;

        let mut base_knn = Knn::new(3, base_train, true);

        // Shuffle once
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..base_knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        base_knn.train_dataframe = base_knn.train_dataframe.take(indices_series.u32()?)?;

        let x = base_knn
            .train_dataframe
            .select(base_knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = base_knn
            .train_dataframe
            .column(base_knn.train_dataframe.get_column_names()[8])?
            .clone();

        let n          = base_knn.train_dataframe.height();
        let num_folds  = 10;
        let fold_size  = n / num_folds;

        let mut total_correct = 0;
        let mut total_count   = 0;
        let mut fold_accuracies: Vec<f32> = Vec::new();

        for fold in 0..num_folds {
            let test_start = fold * fold_size;
            let test_end   = test_start + fold_size;

            let x_test = x.slice(test_start as i64, fold_size);
            let y_test = y.slice(test_start as i64, fold_size);

            let x_train_a = x.slice(0, test_start);
            let x_train_b = x.slice(test_end as i64, n - test_end);
            let mut y_train_a = y.slice(0, test_start);
            let y_train_b = y.slice(test_end as i64, n - test_end);

            let mut x_train = x_train_a.vstack(&x_train_b)?;
            let y_train = y_train_a.extend(&y_train_b)?;

            let stats = Knn::transform(&mut x_train, None)?;
            let mut x_test = x_test;
            Knn::transform(&mut x_test, stats)?;

            let mut knn = base_knn.clone();
            knn.k = 3;

            let mut fold_correct = 0;
            for row_test in 0..x_test.height() {
                let inputs: Vec<AnyValue> = x_test.get_row(row_test)?.0;
                let prediction = knn.predict(&inputs, &x_train, &y_train)?;
                let real = y_test.get(row_test)?.to_string();

                if prediction == real {
                    fold_correct += 1;
                    println!(
                        "{}",
                        &format!("[fold {}] prediction : {}, Real: {}", fold + 1, prediction, real).green()
                    );
                } else {
                    println!(
                        "{}",
                        &format!("[fold {}] prediction : {}, Real: {}", fold + 1, prediction, real).red()
                    );
                }
            }

            let fold_accuracy = fold_correct as f32 / fold_size as f32 * 100.0;
            println!(
                "Fold {}/{} — Accuracy: {}/{} = {:.2}%\n",
                fold + 1, num_folds, fold_correct, fold_size, fold_accuracy
            );

            total_correct += fold_correct;
            total_count   += fold_size;
            fold_accuracies.push(fold_accuracy);
        }

        let mean_accuracy = fold_accuracies.iter().sum::<f32>() / num_folds as f32;
        println!(
            "10-fold CV — Total: {}/{} = {:.2}%",
            total_correct, total_count, mean_accuracy
        );

        Ok(())
    }
}
