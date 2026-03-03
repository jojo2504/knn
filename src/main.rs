use std::{fs::File, io::{BufWriter, Write}};

use anyhow::Ok;
use knn::{distance::DistanceMetric, knn::Knn};
use polars::{io::SerReader, prelude::CsvReadOptions};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn main() -> anyhow::Result<()> {
    let train = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(
            "/home/jojo/Documents/rust/knn/assets/train.csv".into(),
        ))?
        .finish()?
        .drop_nulls::<String>(None)?;

    let input = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(
            "/home/jojo/Documents/rust/knn/assets/test.csv".into(),
        ))?
        .finish()?
        .drop_nulls::<String>(None)?;

    // Best config: k=3, Manhattan, C2*7.0, Gaussian sigma=2.0 → 99.21% LOO
    let mut knn = Knn::new(3, train);
    knn.metric = DistanceMetric::Manhattan;
    knn.sigma = Some(2.0);

    let mut x_train = knn.train_dataframe
        .select(knn.train_dataframe.get_column_names()[1..8].to_vec())?;
    let y_train = knn.train_dataframe.column("Label")?;

    // Normalize train, then normalize test with the same stats
    let stats = Knn::transform(&mut x_train, None)?;

    // Extract test IDs before selecting feature columns
    let test_ids: Vec<String> = input.column("Id")?
        .cast(&polars::prelude::DataType::String)?
        .str()?
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect();
    let mut x_test = input.select(input.get_column_names()[1..8].to_vec())?;
    Knn::transform(&mut x_test, stats)?;

    // Amplify C2 by 7.0 on both sets
    Knn::amplify_features(&mut x_train, &["C2"], 7.0)?;
    Knn::amplify_features(&mut x_test, &["C2"], 7.0)?;

    let results: Vec<String> = (0..x_test.height())
        .into_par_iter()
        .map(|row_test| {
            let input_row = x_test.get_row(row_test).unwrap().0;
            let prediction = knn.predict(&input_row, &x_train, y_train, None).unwrap();
            format!("{},{}", test_ids[row_test], prediction)
        })
        .collect();

    let file = File::create("/home/jojo/Documents/rust/knn/src/sample_submission.csv")?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Id,Label")?;
    writer.write_all(results.join("\n").as_bytes())?;
    writer.flush()?;

    Ok(())
}