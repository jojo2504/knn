pub mod distance;
pub mod knn;

#[cfg(test)]
mod tests {
    use polars::{
        io::SerReader,
        prelude::{AnyValue, CsvReadOptions, NamedFrom},
        series::Series,
    };
    use rand::{rng, seq::SliceRandom};
    use colored::Colorize;

    use crate::distance::DistanceMetric;
    use crate::knn::Knn;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    fn loo_accuracy(
        x: &polars::frame::DataFrame,
        y: &polars::prelude::Column,
        k: u32,
        metric: DistanceMetric,
        class_weights: bool,
        diff_features: bool,
    ) -> anyhow::Result<f32> {
        let n = x.height();
        let correct: u32 = (0..n)
            .into_par_iter()
            .map(|i| {
                let x_train_a = x.slice(0, i);
                let x_train_b = x.slice(i as i64 + 1, n - i - 1);
                let mut y_train_a = y.slice(0, i);
                let y_train_b = y.slice(i as i64 + 1, n - i - 1);
                let mut x_train = x_train_a.vstack(&x_train_b).unwrap();
                let y_train = y_train_a.extend(&y_train_b).unwrap();
                let mut x_test_row = x.slice(i as i64, 1);
                if diff_features {
                    Knn::add_diff_features(&mut x_train).unwrap();
                    Knn::add_diff_features(&mut x_test_row).unwrap();
                }
                let stats = Knn::transform(&mut x_train, None).unwrap();
                Knn::transform(&mut x_test_row, stats).unwrap();
                let cw_map = if class_weights { Some(Knn::compute_class_weights(y_train)) } else { None };
                let mut knn = Knn::new(k, polars::frame::DataFrame::default());
                knn.metric = metric;
                let inputs: Vec<AnyValue> = x_test_row.get_row(0).unwrap().0;
                let pred = knn.predict(&inputs, &x_train, y_train, cw_map.as_ref()).unwrap();
                let real = y.get(i).unwrap().to_string();
                (pred == real) as u32
            })
            .sum();
        Ok(correct as f32 / n as f32 * 100.0)
    }

    fn loo_weighted(
        x: &polars::frame::DataFrame,
        y: &polars::prelude::Column,
        k: u32,
        metric: DistanceMetric,
        fw: &[(&'static str, f64)],
        use_ratios: bool,
    ) -> anyhow::Result<f32> {
        let n = x.height();
        let fw_owned: Vec<(&'static str, f64)> = fw.to_vec();
        let correct: u32 = (0..n)
            .into_par_iter()
            .map(|i| {
                let xa = x.slice(0, i);
                let xb = x.slice(i as i64 + 1, n - i - 1);
                let mut ya = y.slice(0, i);
                let yb = y.slice(i as i64 + 1, n - i - 1);
                let mut xtr = xa.vstack(&xb).unwrap();
                let ytr = ya.extend(&yb).unwrap();
                let mut xte = x.slice(i as i64, 1);
                if use_ratios {
                    Knn::add_ratio_features(&mut xtr).unwrap();
                    Knn::add_ratio_features(&mut xte).unwrap();
                }
                let stats = Knn::transform(&mut xtr, None).unwrap();
                Knn::transform(&mut xte, stats).unwrap();
                for &(col, w) in &fw_owned {
                    Knn::amplify_features(&mut xtr, &[col], w).unwrap();
                    Knn::amplify_features(&mut xte, &[col], w).unwrap();
                }
                let mut knn = Knn::new(k, polars::frame::DataFrame::default());
                knn.metric = metric;
                let inputs: Vec<AnyValue> = xte.get_row(0).unwrap().0;
                let pred = knn.predict(&inputs, &xtr, ytr, None).unwrap();
                let real = y.get(i).unwrap().to_string();
                (pred == real) as u32
            })
            .sum();
        Ok(correct as f32 / n as f32 * 100.0)
    }

    #[test]
    pub fn test_knn_with_train_df() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let mut knn = Knn::new(3, train);
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        knn.train_dataframe = knn.train_dataframe.take(indices_series.u32()?)?;
        let x = knn.train_dataframe.select(knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = knn.train_dataframe.column(knn.train_dataframe.get_column_names()[8])?;
        let n = knn.train_dataframe.height();
        let train_size = (n as f64 * 0.8) as usize;
        let mut x_train = x.slice(0, train_size);
        let y_train = y.slice(0, train_size);
        let mut x_test = x.slice(train_size as i64, n - train_size);
        let y_test = y.slice(train_size as i64, n - train_size);
        let stats = Knn::transform(&mut x_train, None)?;
        Knn::transform(&mut x_test, stats)?;
        let mut correct = 0; let mut total = 0;
        for row_test in 0..x_test.height() {
            let inputs: Vec<AnyValue> = x_test.get_row(row_test)?.0;
            let prediction = knn.predict(&inputs, &x_train, &y_train, None)?;
            let real = y_test.get(row_test)?.to_string();
            if prediction == real { correct += 1; println!("{}", &format!("pred={} real={}", prediction, real).green()); }
            else { println!("{}", &format!("pred={} real={}", prediction, real).red()); }
            total += 1;
            println!("Accuracy: {}/{} = {:.2}%", correct, total, (correct as f32 / total as f32) * 100.0);
        }
        Ok(())
    }

    #[test]
    pub fn find_best_k() -> anyhow::Result<()> {
        let base_train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let mut base_knn = Knn::new(0, base_train.clone());
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..base_knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        base_knn.train_dataframe = base_knn.train_dataframe.take(indices_series.u32()?)?;
        let x = base_knn.train_dataframe.select(base_knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = base_knn.train_dataframe.column(base_knn.train_dataframe.get_column_names()[8])?;
        let n = base_knn.train_dataframe.height();
        let train_size = (n as f64 * 0.8) as usize;
        let mut x_train = x.slice(0, train_size);
        let y_train = y.slice(0, train_size);
        let mut x_test = x.slice(train_size as i64, n - train_size);
        let y_test = y.slice(train_size as i64, n - train_size);
        let stats = Knn::transform(&mut x_train, None)?;
        Knn::transform(&mut x_test, stats)?;
        let mut best_accuracy = 0.0; let mut best_k = 0;
        for k in 2..15 {
            let mut knn = base_knn.clone(); knn.k = k;
            let mut correct = 0; let mut total = 0;
            for row_test in 0..x_test.height() {
                let inputs: Vec<AnyValue> = x_test.get_row(row_test)?.0;
                let prediction = knn.predict(&inputs, &x_train, &y_train, None)?;
                let real = y_test.get(row_test)?.to_string();
                if prediction == real { correct += 1; }
                total += 1;
            }
            let acc = (correct as f32 / total as f32) * 100.0;
            if acc > best_accuracy { best_k = k; best_accuracy = acc; }
            println!("Accuracy: {}/{} = {:.2}% with k={}", correct, total, acc, k);
        }
        println!("best k={} with accuracy={}", best_k, best_accuracy);
        Ok(())
    }

    #[test]
    pub fn find_best_k_fold() -> anyhow::Result<()> {
        let base_train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let mut base_knn = Knn::new(0, base_train.clone());
        let mut rng = rng();
        let mut indices: Vec<u32> = (0..base_knn.train_dataframe.height() as u32).collect();
        indices.shuffle(&mut rng);
        let indices_series = Series::new("idx".into(), indices);
        base_knn.train_dataframe = base_knn.train_dataframe.take(indices_series.u32()?)?;
        let x = base_knn.train_dataframe.select(base_knn.train_dataframe.get_column_names()[1..8].to_vec())?;
        let y = base_knn.train_dataframe.column(base_knn.train_dataframe.get_column_names()[8])?.clone();
        let n = base_knn.train_dataframe.height();
        let num_folds = 5;
        let fold_size = n / num_folds;
        let mut best_acc = 0.0f32; let mut best_k = 0u32;
        let mut best_metric = DistanceMetric::Manhattan;

        let metrics: &[DistanceMetric] = &[
            DistanceMetric::Manhattan,
            DistanceMetric::Euclidean,
            DistanceMetric::Minkowski(3),
        ];

        for &metric in metrics {
            for k in 1u32..=20 {
                let mut fold_accuracies: Vec<f32> = Vec::new();
                for fold in 0..num_folds {
                    let ts = fold * fold_size;
                    let te = ts + fold_size;
                    let y_test = y.slice(ts as i64, fold_size);
                    let xa = x.slice(0, ts);
                    let xb = x.slice(te as i64, n - te);
                    let mut ya = y.slice(0, ts);
                    let yb = y.slice(te as i64, n - te);
                    let mut xtr = xa.vstack(&xb)?;
                    let ytr = ya.extend(&yb)?;
                    let mut xte = x.slice(ts as i64, fold_size);
                    let stats = Knn::transform(&mut xtr, None)?;
                    Knn::transform(&mut xte, stats)?;
                    Knn::amplify_features(&mut xtr, &["C2"], 8.0)?;
                    Knn::amplify_features(&mut xte, &["C2"], 8.0)?;
                    let mut knn = base_knn.clone(); knn.k = k; knn.metric = metric;
                    let mut correct = 0;
                    for row_test in 0..xte.height() {
                        let inputs: Vec<AnyValue> = xte.get_row(row_test)?.0;
                        let prediction = knn.predict(&inputs, &xtr, ytr, None)?;
                        let real = y_test.get(row_test)?.to_string();
                        if prediction == real { correct += 1; }
                    }
                    fold_accuracies.push(correct as f32 / fold_size as f32 * 100.0);
                }
                let mean_acc = fold_accuracies.iter().sum::<f32>() / num_folds as f32;
                if mean_acc > best_acc {
                    best_acc = mean_acc; best_k = k; best_metric = metric;
                    println!("NEW BEST ► k={} metric={:?} → {:.2}%", k, metric, mean_acc);
                }
            }
        }
        println!("\nbest k={} metric={:?} accuracy={:.2}%", best_k, best_metric, best_acc);
        Ok(())
    }

    #[test]
    pub fn find_best_loo() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let metrics = [DistanceMetric::Manhattan, DistanceMetric::Euclidean, DistanceMetric::Minkowski(3)];
        let mut best = (0.0f32, 0u32, DistanceMetric::Manhattan, false, false);
        for &diff in &[false, true] {
            for &cw in &[false, true] {
                for &metric in &metrics {
                    for k in 1u32..=15 {
                        let acc = loo_accuracy(&x, y, k, metric, cw, diff)?;
                        if acc > best.0 {
                            best = (acc, k, metric, cw, diff);
                            println!("NEW BEST ► k={} metric={:?} cw={} diff={} → {:.4}%", k, metric, cw, diff, acc);
                        }
                    }
                }
            }
        }
        println!("\nLOO best: k={} metric={:?} cw={} diff={} acc={:.4}%", best.1, best.2, best.3, best.4, best.0);
        Ok(())
    }

    #[test]
    pub fn find_best_feature_weights() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let mut best_acc = 0.0f32;

        let multipliers: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0];
        for &col in &["C1","C2","C3","C4","C5","C6","C7"] {
            for &mult in multipliers {
                for k in 1u32..=7 {
                    let acc = loo_weighted(&x, y, k, DistanceMetric::Manhattan, &[(col, mult)], false)?;
                    if acc > best_acc { best_acc = acc; println!("NEW BEST ► col={} mult={} k={} → {:.4}%", col, mult, k, acc); }
                }
            }
        }

        let fine_c2: Vec<f64> = (10..=200).map(|v| v as f64 * 0.5).collect();
        for &c2m in &fine_c2 {
            for k in 1u32..=7 {
                let acc = loo_weighted(&x, y, k, DistanceMetric::Manhattan, &[("C2", c2m)], false)?;
                if acc > best_acc { best_acc = acc; println!("NEW BEST fine ► C2*{} k={} → {:.4}%", c2m, k, acc); }
            }
        }

        let mults2: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        for &c2m in mults2 { for &c3m in mults2 {
            let acc = loo_weighted(&x, y, 3, DistanceMetric::Manhattan, &[("C2", c2m), ("C3", c3m)], false)?;
            if acc > best_acc { best_acc = acc; println!("NEW BEST C2+C3 ► C2*{} C3*{} → {:.4}%", c2m, c3m, acc); }
        }}
        for &c1m in mults2 { for &c2m in mults2 {
            let acc = loo_weighted(&x, y, 3, DistanceMetric::Manhattan, &[("C1", c1m), ("C2", c2m)], false)?;
            if acc > best_acc { best_acc = acc; println!("NEW BEST C1+C2 ► C1*{} C2*{} → {:.4}%", c1m, c2m, acc); }
        }}

        for &c2m in &[6.0f64, 8.0, 10.0, 12.0] {
            for k in 1u32..=5 {
                let acc = loo_weighted(&x, y, k, DistanceMetric::Manhattan, &[("C2", c2m)], true)?;
                if acc > best_acc { best_acc = acc; println!("NEW BEST ratios+C2*{} k={} → {:.4}%", c2m, k, acc); }
            }
        }

        println!("\n=== GRAND TOTAL BEST: {:.4}% ===", best_acc);
        Ok(())
    }

    #[test]
    pub fn find_best_gaussian() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let n = x.height();
        let mut best_acc = 0.0f32;

        // Fine sweep around best found: C2*7, sigma=2.0, k=3
        // sigma: None + 0.5..5.0 in 0.1 steps
        let sigma_vals: Vec<Option<f32>> = {
            let mut v: Vec<Option<f32>> = vec![None];
            v.extend((5u32..=60).map(|i| Some(i as f32 * 0.1)));
            v
        };
        // C2: 5.0..9.5 in 0.25 steps
        let c2_mults: Vec<f64> = (20u32..=38).map(|i| i as f64 * 0.25).collect();
        let k_vals: Vec<u32> = (2..=6).collect();

        for &c2m in &c2_mults {
            for &sigma in &sigma_vals {
                for &k in &k_vals {
                    let correct: u32 = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let xa = x.slice(0, i);
                            let xb = x.slice(i as i64 + 1, n - i - 1);
                            let mut ya = y.slice(0, i);
                            let yb = y.slice(i as i64 + 1, n - i - 1);
                            let mut xtr = xa.vstack(&xb).unwrap();
                            let ytr = ya.extend(&yb).unwrap();
                            let mut xte = x.slice(i as i64, 1);
                            let stats = Knn::transform(&mut xtr, None).unwrap();
                            Knn::transform(&mut xte, stats).unwrap();
                            Knn::amplify_features(&mut xtr, &["C2"], c2m).unwrap();
                            Knn::amplify_features(&mut xte, &["C2"], c2m).unwrap();
                            let mut knn = Knn::new(k, polars::frame::DataFrame::default());
                            knn.metric = DistanceMetric::Manhattan;
                            knn.sigma = sigma;
                            let inputs: Vec<AnyValue> = xte.get_row(0).unwrap().0;
                            let pred = knn.predict(&inputs, &xtr, ytr, None).unwrap();
                            let real = y.get(i).unwrap().to_string();
                            (pred == real) as u32
                        })
                        .sum();
                    let acc = correct as f32 / n as f32 * 100.0;
                    if acc > best_acc {
                        best_acc = acc;
                        println!("NEW BEST ► C2*{} sigma={:?} k={} → {:.4}%", c2m, sigma, k, acc);
                    }
                }
            }
        }
        println!("\n=== GAUSSIAN BEST: {:.4}% ===", best_acc);
        Ok(())
    }

    /// Oversamples the 6 hard class-3 outlier samples (those misclassified as class-0)
    /// by duplicating them N times in the training set before each LOO evaluation.
    /// LOO is still measured only on the original 1012 samples.
    #[test]
    pub fn find_best_augmented() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x_full = train.select(train.get_column_names()[1..8].to_vec())?;
        let y_full = train.column(train.get_column_names()[8])?;
        let n = x_full.height();

        // Hard class-3 → class-0 boundary outliers (0-based indices)
        let hard_indices: &[usize] = &[206, 482, 540, 624, 760, 856];

        // Pre-extract immutable row slices (shared safely across Rayon threads)
        let hard_x_rows: Vec<polars::frame::DataFrame> = hard_indices.iter()
            .map(|&i| x_full.slice(i as i64, 1))
            .collect();
        let hard_y_rows: Vec<polars::prelude::Column> = hard_indices.iter()
            .map(|&i| y_full.slice(i as i64, 1))
            .collect();
        let y_strings: Vec<String> = (0..n)
            .map(|i| y_full.get(i).unwrap().to_string())
            .collect();

        let mut best_acc = 0.0f32;

        for &n_copies in &[1usize, 2, 3, 5, 10, 20, 50] {
            for &c2m in &[7.0f64, 8.0, 9.0, 10.0] {
                for k in 1u32..=5 {
                    let correct: u32 = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let xa = x_full.slice(0, i);
                            let xb = x_full.slice(i as i64 + 1, n - i - 1);
                            let mut ya = y_full.slice(0, i);
                            let yb = y_full.slice(i as i64 + 1, n - i - 1);
                            ya.extend(&yb).unwrap();
                            let mut xtr = xa.vstack(&xb).unwrap();
                            // Append n_copies of each hard sample (skip current sample i)
                            for (hi, &hard_idx) in hard_indices.iter().enumerate() {
                                if hard_idx == i { continue; }
                                for _ in 0..n_copies {
                                    xtr.extend(&hard_x_rows[hi]).unwrap();
                                    ya.extend(&hard_y_rows[hi]).unwrap();
                                }
                            }
                            let mut xte = x_full.slice(i as i64, 1);
                            let stats = Knn::transform(&mut xtr, None).unwrap();
                            Knn::transform(&mut xte, stats).unwrap();
                            Knn::amplify_features(&mut xtr, &["C2"], c2m).unwrap();
                            Knn::amplify_features(&mut xte, &["C2"], c2m).unwrap();
                            let mut knn = Knn::new(k, polars::frame::DataFrame::default());
                            knn.metric = DistanceMetric::Manhattan;
                            let inputs: Vec<AnyValue> = xte.get_row(0).unwrap().0;
                            let pred = knn.predict(&inputs, &xtr, &ya, None).unwrap();
                            (pred == y_strings[i]) as u32
                        })
                        .sum();
                    let acc = correct as f32 / n as f32 * 100.0;
                    if acc > best_acc {
                        best_acc = acc;
                        println!("NEW BEST ► n_copies={} c2m={} k={} → {:.4}%",
                            n_copies, c2m, k, acc);
                    }
                }
            }
        }
        println!("\n=== AUGMENTED BEST: {:.4}% ===", best_acc);
        Ok(())
    }

    /// Combined: oversample hard class-3 outliers + Gaussian kernel sweep
    #[test]
    pub fn find_best_combined() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x_full = train.select(train.get_column_names()[1..8].to_vec())?;
        let y_full = train.column(train.get_column_names()[8])?;
        let n = x_full.height();

        let hard_indices: &[usize] = &[206, 482, 540, 624, 760, 856];
        let hard_x_rows: Vec<polars::frame::DataFrame> = hard_indices.iter()
            .map(|&i| x_full.slice(i as i64, 1)).collect();
        let hard_y_rows: Vec<polars::prelude::Column> = hard_indices.iter()
            .map(|&i| y_full.slice(i as i64, 1)).collect();
        let y_strings: Vec<String> = (0..n)
            .map(|i| y_full.get(i).unwrap().to_string()).collect();

        let mut best_acc = 0.0f32;

        // sigma: None + 0.5..5.0 in 0.25 steps
        let sigma_vals: Vec<Option<f32>> = {
            let mut v = vec![None];
            v.extend((2u32..=20).map(|i| Some(i as f32 * 0.25)));
            v
        };

        for &n_copies in &[1usize, 2, 3, 5, 10] {
            for &c2m in &[6.0f64, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0] {
                for &sigma in &sigma_vals {
                    for k in 2u32..=5 {
                        let correct: u32 = (0..n)
                            .into_par_iter()
                            .map(|i| {
                                let xa = x_full.slice(0, i);
                                let xb = x_full.slice(i as i64 + 1, n - i - 1);
                                let mut ya = y_full.slice(0, i);
                                let yb = y_full.slice(i as i64 + 1, n - i - 1);
                                ya.extend(&yb).unwrap();
                                let mut xtr = xa.vstack(&xb).unwrap();
                                for (hi, &hard_idx) in hard_indices.iter().enumerate() {
                                    if hard_idx == i { continue; }
                                    for _ in 0..n_copies {
                                        xtr.extend(&hard_x_rows[hi]).unwrap();
                                        ya.extend(&hard_y_rows[hi]).unwrap();
                                    }
                                }
                                let mut xte = x_full.slice(i as i64, 1);
                                let stats = Knn::transform(&mut xtr, None).unwrap();
                                Knn::transform(&mut xte, stats).unwrap();
                                Knn::amplify_features(&mut xtr, &["C2"], c2m).unwrap();
                                Knn::amplify_features(&mut xte, &["C2"], c2m).unwrap();
                                let mut knn = Knn::new(k, polars::frame::DataFrame::default());
                                knn.metric = DistanceMetric::Manhattan;
                                knn.sigma = sigma;
                                let inputs: Vec<AnyValue> = xte.get_row(0).unwrap().0;
                                let pred = knn.predict(&inputs, &xtr, &ya, None).unwrap();
                                (pred == y_strings[i]) as u32
                            })
                            .sum();
                        let acc = correct as f32 / n as f32 * 100.0;
                        if acc > best_acc {
                            best_acc = acc;
                            println!("NEW BEST ► n_copies={} c2m={} sigma={:?} k={} → {:.4}%",
                                n_copies, c2m, sigma, k, acc);
                        }
                    }
                }
            }
        }
        println!("\n=== COMBINED BEST: {:.4}% ===", best_acc);
        Ok(())
    }

    /// Sweep C2*c2m + secondary_feature*w with Gaussian kernel
    #[test]
    pub fn find_best_multi_feature() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x_full = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let n = x_full.height();
        let mut best_acc = 0.0f32;

        let c2_mults  = [6.0f64, 6.5, 7.0, 7.5, 8.0];
        let sec_feats = ["C1", "C3", "C4", "C5", "C6", "C7"];
        let sec_mults = [0.3f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0];
        let sigmas: &[Option<f32>] = &[None, Some(1.5), Some(2.0), Some(2.5), Some(3.0)];
        let k_vals: &[u32] = &[2, 3, 4, 5];

        for &c2m in &c2_mults {
            for &feat in &sec_feats {
                for &mult in &sec_mults {
                    for &sigma in sigmas {
                        for &k in k_vals {
                            let correct: u32 = (0..n)
                                .into_par_iter()
                                .map(|i| {
                                    let xa = x_full.slice(0, i);
                                    let xb = x_full.slice(i as i64 + 1, n - i - 1);
                                    let mut ya = y.slice(0, i);
                                    let yb = y.slice(i as i64 + 1, n - i - 1);
                                    let mut xtr = xa.vstack(&xb).unwrap();
                                    let ytr = ya.extend(&yb).unwrap();
                                    let mut xte = x_full.slice(i as i64, 1);
                                    let stats = Knn::transform(&mut xtr, None).unwrap();
                                    Knn::transform(&mut xte, stats).unwrap();
                                    Knn::amplify_features(&mut xtr, &["C2"], c2m).unwrap();
                                    Knn::amplify_features(&mut xte, &["C2"], c2m).unwrap();
                                    if mult != 1.0 {
                                        Knn::amplify_features(&mut xtr, &[feat], mult).unwrap();
                                        Knn::amplify_features(&mut xte, &[feat], mult).unwrap();
                                    }
                                    let mut knn = Knn::new(k, polars::frame::DataFrame::default());
                                    knn.metric = DistanceMetric::Manhattan;
                                    knn.sigma = sigma;
                                    let inputs: Vec<AnyValue> = xte.get_row(0).unwrap().0;
                                    let pred = knn.predict(&inputs, &xtr, ytr, None).unwrap();
                                    let real = y.get(i).unwrap().to_string();
                                    (pred == real) as u32
                                })
                                .sum();
                            let acc = correct as f32 / n as f32 * 100.0;
                            if acc > best_acc {
                                best_acc = acc;
                                println!("NEW BEST ► C2*{} {}*{} sigma={:?} k={} → {:.4}%",
                                    c2m, feat, mult, sigma, k, acc);
                            }
                        }
                    }
                }
            }
        }
        println!("\n=== MULTI-FEAT BEST: {:.4}% ===", best_acc);
        Ok(())
    }

    /// Confusion matrix with best Gaussian config: C2*7, sigma=2.0, k=3
    #[test]
    pub fn confusion_matrix_gaussian() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x_full = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let n = x_full.height();
        let mut errors: Vec<(usize, String, String)> = Vec::new();
        for i in 0..n {
            let xa = x_full.slice(0, i);
            let xb = x_full.slice(i as i64 + 1, n - i - 1);
            let mut ya = y.slice(0, i);
            let yb = y.slice(i as i64 + 1, n - i - 1);
            let mut xtr = xa.vstack(&xb)?;
            let ytr = ya.extend(&yb)?;
            let mut xte = x_full.slice(i as i64, 1);
            let stats = Knn::transform(&mut xtr, None)?;
            Knn::transform(&mut xte, stats)?;
            Knn::amplify_features(&mut xtr, &["C2"], 7.0)?;
            Knn::amplify_features(&mut xte, &["C2"], 7.0)?;
            let mut knn = Knn::new(3, polars::frame::DataFrame::default());
            knn.metric = DistanceMetric::Manhattan;
            knn.sigma = Some(2.0);
            let inputs: Vec<AnyValue> = xte.get_row(0)?.0;
            let pred = knn.predict(&inputs, &xtr, ytr, None)?;
            let real = y.get(i)?.to_string();
            if pred != real { errors.push((i, real, pred)); }
        }
        println!("\nTotal errors: {}/{} ({:.4}% accuracy)", errors.len(), n,
            (n - errors.len()) as f32 / n as f32 * 100.0);
        let feature_names = ["C1","C2","C3","C4","C5","C6","C7"];
        for (i, real, pred) in &errors {
            print!("  sample={:4} real={} pred={} ", i, real, pred);
            for &f in &feature_names {
                let val = train.column(f)?.get(*i)?.try_extract::<f64>()?;
                print!(" {}={:.2}", f, val);
            }
            println!();
        }
        Ok(())
    }

    #[test]
    pub fn confusion_matrix_best() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x_full = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let n = x_full.height();
        let mut errors: Vec<(usize, String, String)> = Vec::new();
        for i in 0..n {
            let xa = x_full.slice(0, i);
            let xb = x_full.slice(i as i64 + 1, n - i - 1);
            let mut ya = y.slice(0, i);
            let yb = y.slice(i as i64 + 1, n - i - 1);
            let mut xtr = xa.vstack(&xb)?;
            let ytr = ya.extend(&yb)?;
            let mut xte = x_full.slice(i as i64, 1);
            let stats = Knn::transform(&mut xtr, None)?;
            Knn::transform(&mut xte, stats)?;
            Knn::amplify_features(&mut xtr, &["C2"], 8.0)?;
            Knn::amplify_features(&mut xte, &["C2"], 8.0)?;
            let mut knn = Knn::new(3, polars::frame::DataFrame::default());
            knn.metric = DistanceMetric::Manhattan;
            let inputs: Vec<AnyValue> = xte.get_row(0)?.0;
            let pred = knn.predict(&inputs, &xtr, ytr, None)?;
            let real = y.get(i)?.to_string();
            if pred != real { errors.push((i, real, pred)); }
        }
        println!("\nTotal errors: {}/{} ({:.4}% accuracy)", errors.len(), n,
            (n - errors.len()) as f32 / n as f32 * 100.0);
        let feature_names = ["C1","C2","C3","C4","C5","C6","C7"];
        for (i, real, pred) in &errors {
            print!("  sample={:4} real={} pred={} ", i, real, pred);
            for &f in &feature_names {
                let val = train.column(f)?.get(*i)?.try_extract::<f64>()?;
                print!(" {}={:.2}", f, val);
            }
            println!();
        }
        Ok(())
    }

    #[test]
    pub fn analyze_hard_errors() -> anyhow::Result<()> {
        let train = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("/home/jojo/Documents/rust/knn/assets/train.csv".into()))?
            .finish()?.drop_nulls::<String>(None)?;
        let x_full = train.select(train.get_column_names()[1..8].to_vec())?;
        let y = train.column(train.get_column_names()[8])?;
        let n = x_full.height();
        let hard_errors = [206usize, 226, 482, 540, 624, 665, 738, 760, 856];
        for &target in &hard_errors {
            println!("\n=== Sample {} (true label={}) ===", target, y.get(target)?.to_string());
            let xa = x_full.slice(0, target);
            let xb = x_full.slice(target as i64 + 1, n - target - 1);
            let mut ya = y.slice(0, target);
            let yb = y.slice(target as i64 + 1, n - target - 1);
            let mut xtr = xa.vstack(&xb)?;
            let ytr = ya.extend(&yb)?;
            let mut xte = x_full.slice(target as i64, 1);
            let stats = Knn::transform(&mut xtr, None)?;
            Knn::transform(&mut xte, stats)?;
            Knn::amplify_features(&mut xtr, &["C2"], 8.0)?;
            Knn::amplify_features(&mut xte, &["C2"], 8.0)?;
            let test_row: Vec<AnyValue> = xte.get_row(0)?.0;
            let mut dists: Vec<(f32, usize, String)> = (0..xtr.height())
                .map(|j| {
                    use crate::distance::{distance, DistanceMetric};
                    let train_row = xtr.get_row(j).unwrap().0;
                    let d = distance(&test_row, &train_row, DistanceMetric::Manhattan);
                    let orig_idx = if j < target { j } else { j + 1 };
                    let class = ytr.get(j).unwrap().to_string();
                    (d, orig_idx, class)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            println!("  Top 10 nearest neighbors:");
            for (d, idx, cls) in dists.iter().take(10) {
                println!("    idx={:4} dist={:.6} label={}", idx, d, cls);
            }
        }
        Ok(())
    }
}
