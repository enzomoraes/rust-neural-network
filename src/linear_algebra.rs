use core::fmt;
use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: MatrixData,
}

pub type MatrixData = Vec<Vec<f64>>;

impl Matrix {
    pub fn new(data: MatrixData) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }
    pub fn zero(rows: usize, cols: usize) -> Matrix {
        let data: MatrixData = vec![vec![0.0; cols]; rows];
        return Matrix::new(data);
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut matrix = Matrix::zero(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                matrix.data[i][j] = thread_rng().gen::<f64>() * 2.0 - 1.0;
            }
        }
        return matrix;
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposed_matrix = Matrix::zero(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_matrix.data[j][i] = self.data[i][j]
            }
        }
        return transposed_matrix;
    }

    pub fn add(&self, matrix_b: &Matrix) -> Matrix {
        if self.rows != matrix_b.rows || self.cols != matrix_b.cols {
            panic!("Cannot add matrices. {}x{} & {}x{}", self.rows, self.cols, matrix_b.rows, matrix_b.cols)
        };

        let mut added_matrix: Matrix = Matrix::zero(self.rows, matrix_b.cols);
        for i in 0..added_matrix.rows {
            for j in 0..added_matrix.cols {
                added_matrix.data[i][j] = self.data[i][j] + matrix_b.data[i][j];
            }
        }
        return added_matrix;
    }

    pub fn subtract(&self, matrix_b: &Matrix) -> Matrix {
        if self.rows != matrix_b.rows || self.cols != matrix_b.cols {
            panic!("Cannot subtract matrices.")
        };

        let mut subtracted_matrix: Matrix = Matrix::zero(self.rows, matrix_b.cols);
        for i in 0..subtracted_matrix.rows {
            for j in 0..subtracted_matrix.cols {
                subtracted_matrix.data[i][j] = self.data[i][j] - matrix_b.data[i][j];
            }
        }
        return subtracted_matrix;
    }

    pub fn multiply(&self, matrix_b: &Matrix) -> Matrix {
        if self.cols != matrix_b.rows {
            panic!("Cannot multiply matrices.")
        };

        let mut multiplied_matrix: Matrix = Matrix::zero(self.rows, matrix_b.cols);
        for i in 0..self.rows {
            for j in 0..matrix_b.cols {
                let mut sum: f64 = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * matrix_b.data[k][j];
                }
                multiplied_matrix.data[i][j] = sum;
            }
        }
        return multiplied_matrix;
    }

    pub fn hadamard_product(&self, matrix_b: &Matrix) -> Matrix {
        if self.rows != matrix_b.rows || self.cols != matrix_b.cols {
            panic!("Cannot apply hadamard product to matrices.")
        };

        let mut multiplied_matrix: Matrix = Matrix::zero(self.rows, matrix_b.cols);
        for i in 0..multiplied_matrix.rows {
            for j in 0..multiplied_matrix.cols {
                multiplied_matrix.data[i][j] = self.data[i][j] * matrix_b.data[i][j];
            }
        }
        return multiplied_matrix;
    }

    pub fn apply_function(&self, function: &dyn Fn(f64) -> f64) -> Matrix {
        return Matrix::new(
            self.data
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| function(value)).collect())
                .collect(),
        );
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Find the maximum width of any element in the matrix
        let max_width = self.data.iter().flatten().map(|&x| x.to_string().len()).max().unwrap_or(0);

        for i in 0..self.rows {
            write!(f, "|")?;
            for j in 0..self.cols {
                // std::fmt fill/alignment
                let cell_str = format!("{:^width$}", self.data[i][j], width = max_width);
                write!(f, "{}", cell_str)?;
                if j < self.cols - 1 {
                    // Print a whitespace between values in the same row
                    write!(f, "  ")?;
                }
            }
            writeln!(f, "|")?;
        }
        return write!(f, "");
    }
}
