use rand::Rng;
mod linear_algebra_tests;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;

fn get_block_size() -> usize {
    env::var("BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn random(rows: usize, cols: usize, activation: &str) -> Matrix {
        let mut buffer = Vec::<f32>::with_capacity(rows * cols);

        // Define o limite baseado na função de ativação matemática ideal
        let limit = match activation {
            "RELU" => {
                // He Initialization (Ideal para ReLU)
                (6.0 / cols as f32).sqrt()
            }
            "SIGMOID" | "TANH" => {
                // Xavier/Glorot Initialization (Ideal para Sigmoid e Tanh)
                (6.0 / (cols + rows) as f32).sqrt()
            }
            _ => {
                // Fallback padrão se não houver regra específica
                1.0
            }
        };

        for _ in 0..rows * cols {
            // Gera o número dentro do limite calculado perfeitamente
            let num = (rand::thread_rng().gen::<f32>() * 2.0 - 1.0) * limit;
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

        let block_size = get_block_size();
        let rows = self.rows;
        let cols = self.cols;

        let mut result_data = vec![0.0; rows * cols];

        for ii in (0..rows).step_by(block_size) {
            for jj in (0..cols).step_by(block_size) {
                let i_max = (ii + block_size).min(rows);
                let j_max = (jj + block_size).min(cols);

                for i in ii..i_max {
                    for j in jj..j_max {
                        let idx = i * cols + j;
                        result_data[idx] = self.data[idx] + other.data[idx];
                    }
                }
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Cannot subtract matrices with different dimensions"
        );

        let block_size = get_block_size();
        let rows = self.rows;
        let cols = self.cols;

        let mut result_data = vec![0.0; rows * cols];

        for ii in (0..rows).step_by(block_size) {
            for jj in (0..cols).step_by(block_size) {
                let i_max = (ii + block_size).min(rows);
                let j_max = (jj + block_size).min(cols);

                for i in ii..i_max {
                    for j in jj..j_max {
                        let idx = i * cols + j;
                        result_data[idx] = self.data[idx] - other.data[idx];
                    }
                }
            }
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!(
                "Cannot multiply matrices. {}x{} & {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
        }

        let block_size = get_block_size();
        let mut result_data = vec![0.0; self.rows * other.cols];
        let n = self.rows;
        let m = self.cols;
        let p = other.cols;

        for ii in (0..n).step_by(block_size) {
            for jj in (0..p).step_by(block_size) {
                for kk in (0..m).step_by(block_size) {
                    // para cada bloco
                    let i_max = (ii + block_size).min(n);
                    let j_max = (jj + block_size).min(p);
                    let k_max = (kk + block_size).min(m);

                    for i in ii..i_max {
                        for j in jj..j_max {
                            let mut sum = result_data[i * p + j];

                            for k in kk..k_max {
                                sum += self.data[i * m + k] * other.data[k * p + j];
                            }

                            result_data[i * p + j] = sum;
                        }
                    }
                }
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        }
    }

    pub fn hadamard_product(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Cannot apply hadamard product to matrices. {}x{} & {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
        }

        let mut result_data = vec![0.0; self.rows * self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i * self.cols + j] =
                    self.data[i * self.cols + j] * other.data[i * self.cols + j];
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
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

    pub fn identity(size: usize) -> Matrix {
        let mut result_data = vec![0.0; size * size];

        for i in 0..size {
            for j in 0..size {
                if j.eq(&i) {
                    result_data[i * size + j] = 1.0;
                }
            }
        }

        Matrix {
            rows: size,
            cols: size,
            data: result_data,
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
