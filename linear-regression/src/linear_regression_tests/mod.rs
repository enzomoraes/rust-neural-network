#[cfg(test)]
mod linear_regression {
    use crate::{least_mean_squared_error, Matrix};

    #[test]
    fn least_mean_squared_error_test() {
        let predicted: Matrix = Matrix::new(1, 4, vec![1.0, 1.5, 2.0, 2.5]);
        let target: Matrix = Matrix::new(1, 4, vec![2.0, 2.5, 3.0, 3.5]);

        let result = least_mean_squared_error(predicted, target);

        assert_eq!(result, 0.5);
    }

    #[test]
    fn least_mean_squared_error_test2() {
        let predicted: Matrix = Matrix::new(1, 3, vec![0.5, 1.0, 1.5]);
        let target: Matrix = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);

        let result = least_mean_squared_error(predicted, target);

        assert_eq!(result, 0.5833333);
    }
}
