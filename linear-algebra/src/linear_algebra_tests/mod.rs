#[cfg(test)]
mod linear_algebra_tests {
    use crate::Matrix;

    #[test]
    fn matrix_creation() {
        let matrix: Matrix = Matrix::new(2, 2, vec![1.0, 1.5, 2.0, 2.5]);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);

        assert_eq!(matrix.data[0], 1.0);
        assert_eq!(matrix.data[1], 1.5);

        assert_eq!(matrix.data[2], 2.0);
        assert_eq!(matrix.data[3], 2.5);
    }

    #[test]
    fn matrix_of_zeros() {
        let matrix: Matrix = Matrix::zeros(2, 2);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);

        assert_eq!(matrix.data[0], 0.0);
        assert_eq!(matrix.data[1], 0.0);

        assert_eq!(matrix.data[2], 0.0);
        assert_eq!(matrix.data[3], 0.0);
    }

    #[test]
    fn random_matrix() {
        let matrix = Matrix::random(10, 10);

        for i in 0..matrix.rows {
            assert_ne!(matrix.data[i], 0.0);
        }

        assert_eq!(matrix.cols, 10);
        assert_eq!(matrix.rows, 10);
    }

    #[test]
    #[should_panic]
    fn matrix_multiplication_should_fail_when_matrix_a_rows_does_not_match_matrix_b_cols() {
        let a = vec![1.0, 10.0, 4.0, 4.0, 9.0, 7.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 3, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        matrix_a.multiply(&matrix_b);
    }

    #[test]
    fn matrix_2x2_multiplication() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 2, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        let multiplied_matrix: Matrix = matrix_a.multiply(&matrix_b);

        assert_eq!(multiplied_matrix.rows, 2);
        assert_eq!(multiplied_matrix.cols, 2);

        assert_eq!(multiplied_matrix.data[0], 53.0);
        assert_eq!(multiplied_matrix.data[1], 75.0);

        assert_eq!(multiplied_matrix.data[2], 57.0);
        assert_eq!(multiplied_matrix.data[3], 83.0);
    }

    #[test]
    fn matrix_2x1_multiplication() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let b = vec![5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 2, a);
        let matrix_b: Matrix = Matrix::new(2, 1, b);
        let multiplied_matrix: Matrix = matrix_a.multiply(&matrix_b);

        assert_eq!(multiplied_matrix.rows, 2);
        assert_eq!(multiplied_matrix.cols, 1);

        assert_eq!(multiplied_matrix.data[0], 75.0);
        assert_eq!(multiplied_matrix.data[1], 83.0);
    }

    #[test]
    #[should_panic]
    fn matrix_addition_should_fail_when_matrices_are_not_same_dimensions() {
        let a = vec![1.0, 4.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 1, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        matrix_a.add(&matrix_b);
    }

    #[test]
    fn matrix_2x2_adition() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 2, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        let added_matrix: Matrix = matrix_a.add(&matrix_b);

        assert_eq!(added_matrix.rows, 2);
        assert_eq!(added_matrix.cols, 2);

        assert_eq!(added_matrix.data[0], 4.0);
        assert_eq!(added_matrix.data[1], 15.0);

        assert_eq!(added_matrix.data[2], 9.0);
        assert_eq!(added_matrix.data[3], 16.0);
    }

    #[test]
    fn matrix_2x1_adition() {
        let a = vec![1.0, 4.0];
        let b = vec![3.0, 5.0];
        let matrix_a: Matrix = Matrix::new(2, 1, a);
        let matrix_b: Matrix = Matrix::new(2, 1, b);
        let added_matrix: Matrix = matrix_a.add(&matrix_b);

        assert_eq!(added_matrix.rows, 2);
        assert_eq!(added_matrix.cols, 1);

        assert_eq!(added_matrix.data[0], 4.0);
        assert_eq!(added_matrix.data[1], 9.0);
    }

    #[test]
    #[should_panic]
    fn matrix_subtraction_should_fail_when_matrices_are_not_same_dimensions() {
        let a = vec![1.0, 4.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 1, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        matrix_a.subtract(&matrix_b);
    }

    #[test]
    fn matrix_2x2_subtraction() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 2, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        let added_matrix: Matrix = matrix_a.subtract(&matrix_b);

        assert_eq!(added_matrix.rows, 2);
        assert_eq!(added_matrix.cols, 2);

        assert_eq!(added_matrix.data[0], -2.0);
        assert_eq!(added_matrix.data[1], 5.0);

        assert_eq!(added_matrix.data[2], -1.0);
        assert_eq!(added_matrix.data[3], 2.0);
    }

    #[test]
    fn matrix_1x1_subtraction() {
        let a = vec![1.0, 4.0];
        let b = vec![3.0, 5.0];
        let matrix_a: Matrix = Matrix::new(2, 1, a);
        let matrix_b: Matrix = Matrix::new(2, 1, b);
        let added_matrix: Matrix = matrix_a.subtract(&matrix_b);

        assert_eq!(added_matrix.rows, 2);
        assert_eq!(added_matrix.cols, 1);

        assert_eq!(added_matrix.data[0], -2.0);
        assert_eq!(added_matrix.data[1], -1.0);
    }

    #[test]
    #[should_panic]
    fn hadamard_product_should_fail_when_matrices_are_not_same_dimensions() {
        let a = vec![1.0, 4.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 1, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        matrix_a.hadamard_product(&matrix_b);
    }

    #[test]
    fn matrix_2x2_hadamard() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let b = vec![3.0, 5.0, 5.0, 7.0];
        let matrix_a: Matrix = Matrix::new(2, 2, a);
        let matrix_b: Matrix = Matrix::new(2, 2, b);
        let hadamard_product_matrix: Matrix = matrix_a.hadamard_product(&matrix_b);

        assert_eq!(hadamard_product_matrix.rows, 2);
        assert_eq!(hadamard_product_matrix.cols, 2);

        assert_eq!(hadamard_product_matrix.data[0], 3.0);
        assert_eq!(hadamard_product_matrix.data[1], 50.0);

        assert_eq!(hadamard_product_matrix.data[2], 20.0);
        assert_eq!(hadamard_product_matrix.data[3], 63.0);
    }

    #[test]
    fn matrix_1x1_hadamard_product() {
        let a = vec![1.0, 4.0];
        let b = vec![3.0, 5.0];
        let matrix_a: Matrix = Matrix::new(2, 1, a);
        let matrix_b: Matrix = Matrix::new(2, 1, b);
        let hadamard_product_matrix: Matrix = matrix_a.hadamard_product(&matrix_b);

        assert_eq!(hadamard_product_matrix.rows, 2);
        assert_eq!(hadamard_product_matrix.cols, 1);

        assert_eq!(hadamard_product_matrix.data[0], 3.0);
        assert_eq!(hadamard_product_matrix.data[1], 20.0);
    }

    #[test]
    fn transpose_matrix() {
        let a = vec![1.0, 10.0, 5.0, 4.0, 9.0, 3.0];
        let matrix_a: Matrix = Matrix::new(2, 3, a);
        let transposed_matrix: Matrix = matrix_a.transpose();

        assert_eq!(transposed_matrix.rows, 3);
        assert_eq!(transposed_matrix.cols, 2);

        assert_eq!(transposed_matrix.data[0], 1.0);
        assert_eq!(transposed_matrix.data[1], 4.0);

        assert_eq!(transposed_matrix.data[2], 10.0);
        assert_eq!(transposed_matrix.data[3], 9.0);

        assert_eq!(transposed_matrix.data[4], 5.0);
        assert_eq!(transposed_matrix.data[5], 3.0);
    }

    #[test]
    fn apply_function_to_matrix() {
        let a = vec![1.0, 10.0, 4.0, 9.0];
        let matrix_a: Matrix = Matrix::new(2, 2, a);
        let multiplied_by_2_matrix: Matrix = matrix_a.apply_function(&|x| (x * 2.0));

        assert_eq!(multiplied_by_2_matrix.rows, 2);
        assert_eq!(multiplied_by_2_matrix.cols, 2);

        assert_eq!(multiplied_by_2_matrix.data[0], 2.0);
        assert_eq!(multiplied_by_2_matrix.data[1], 20.0);

        assert_eq!(multiplied_by_2_matrix.data[2], 8.0);
        assert_eq!(multiplied_by_2_matrix.data[3], 18.0);
    }
}
