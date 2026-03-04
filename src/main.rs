use std::{env, fs::File, io::{BufWriter, Write}, path::Path};

use anyhow::Ok;
use knn::knn::Knn;
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

    let knn = Knn::new(3, train);

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