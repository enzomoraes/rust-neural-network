use linear_algebra::Matrix;
mod linear_regression_tests;


/// 1/2m * Σi→m(h(x(i)) - y(i))²
fn least_mean_squared_error(predicted: Matrix, target: Matrix) -> f32 {
    let squared_error: Matrix = predicted.subtract(&target).apply_function(&|x| (x * x));
    let sum_squared_error: f32 = squared_error.data.iter().fold(0f32, &|acc, x| (acc + x));

    let total_elements: f32 = (predicted.cols * predicted.rows) as f32;

    return sum_squared_error / (2.0 * total_elements);
}
