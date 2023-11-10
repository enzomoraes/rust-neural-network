use rand::Rng;
mod linear_algebra_tests;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn hadamard_product(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Cannot apply hadamard product to matrices. {}x{} & {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
        }

        let mut result_data = vec![0.0; self.cols * self.rows];
        for i in 0..self.data.len() {
            // double check this
            result_data[i] = self.data[i] * other.data[i]
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut buffer = Vec::<f32>::with_capacity(rows * cols);

        for _ in 0..rows * cols {
            let num = rand::thread_rng().gen::<f32>() * 2.0 - 1.0;

            buffer.push(num);
        }

        Matrix {
            rows,
            cols,
            data: buffer,
        }
    }

    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        assert!(data.len() - 1 != rows * cols, "Invalid Size");
        Matrix { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; cols * rows],
        }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Cannot add matrices. {}x{} & {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
        }

        let mut buffer = Vec::<f32>::with_capacity(self.rows * self.cols);

        for i in 0..self.data.len() {
            let result = self.data[i] + other.data[i];

            buffer.push(result);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Cannot subtract matrices with different dimensions"
        );

        let mut buffer = Vec::<f32>::with_capacity(self.rows * self.cols);

        for i in 0..self.data.len() {
            let result = self.data[i] - other.data[i];

            buffer.push(result);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!(
                "Cannot multiply matrices. {}x{} & {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
        }

        let mut result_data = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result_data[i * other.cols + j] = sum;
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut buffer = vec![0.0; self.cols * self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                buffer[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: buffer,
        }
    }

    pub fn apply_function(&self, func: &dyn Fn(f32) -> f32) -> Matrix {
        let a: Vec<f32> = self.data.iter().map(|&val| func(val)).collect();
        return Matrix {
            cols: self.cols,
            rows: self.rows,
            data: a,
        };
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let max_width = self
            .data
            .iter()
            .map(|f| f.to_string().len())
            .max()
            .unwrap_or(0);
          
        for i in 0..self.rows {
            write!(f, "|")?;
            for j in 0..self.cols {
                // std::fmt fill/alignment
                let cell_str = format!(
                    "{:^width$}",
                    self.data[i * self.cols + j],
                    width = max_width
                );
                write!(f, "{}", cell_str)?;
                if j < self.cols - 1 {
                    // Print a whitespace between values in the same row
                    write!(f, "  ")?;
                }
            }
            writeln!(f, "|")?;
        }
        write!(f, "");
        Ok(())
    }
}

impl From<Vec<f32>> for Matrix {
    /// This method will always return a matrix with rows = vec.len() and cols = 1
    fn from(vec: Vec<f32>) -> Self {
        let rows = vec.len();
        let cols = 1;
        Matrix {
            rows,
            cols,
            data: vec,
        }
    }
}
