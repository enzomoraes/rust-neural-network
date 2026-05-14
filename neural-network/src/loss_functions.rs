use linear_algebra::Matrix;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum LossFunction {
    SquaredError,
}

impl LossFunction {
    pub fn compute_loss(&self, target: &Matrix, predictions: &Matrix) -> Matrix {
        match self {
            LossFunction::SquaredError => {
                return target.subtract(&predictions).apply_function(&|x| x * x);
            }
        }
    }

    pub fn compute_gradient(&self, target: &Matrix, predictions: &Matrix) -> Matrix {
        match self {
            LossFunction::SquaredError => {
                return predictions.subtract(&target).apply_function(&|x| x * 2.0);
            }
        }
    }
}
