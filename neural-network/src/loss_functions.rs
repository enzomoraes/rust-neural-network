use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum LossFunction {
    SquaredError,
}

impl LossFunction {
    /// Computes the raw error between the network's predictions and the actual targets.
    ///
    /// The Loss Function acts as the "judge" of the network, quantifying how far off
    /// the predictions are. This metric is primarily used for evaluating the network's
    /// performance during training and testing.
    ///
    /// # Mathematical Theory (Squared Error)
    /// Calculates the element-wise squared difference between the Target (Y) and
    /// the Prediction (Y_hat).
    ///
    /// Formula: L = (Y - Y_hat)^2
    ///
    /// Squaring the error serves two purposes:
    /// 1. It ensures all errors are positive (so underestimating and overestimating
    ///    both penalize the network).
    /// 2. It heavily penalizes large errors while being forgiving of very small ones,
    ///    creating a convex "bowl" shape that is easy for Gradient Descent to navigate.
    pub fn compute_loss(&self, target: &Array2<f32>, predictions: &Array2<f32>) -> Array2<f32> {
        match self {
            LossFunction::SquaredError => (target - predictions).mapv(|x| x * x),
        }
    }

    /// Computes the derivative of the loss function with respect to the predictions.
    ///
    /// This is the spark that starts the entire Backpropagation process. It tells the
    /// last layer of the network exactly how much the final output needs to change
    /// to move closer to the target.
    ///
    /// # Mathematical Theory (Squared Error Gradient)
    /// We need the derivative of the Loss (L) with respect to the prediction (Y_hat).
    /// Applying the Chain Rule to L = (Y - Y_hat)^2:
    ///
    /// 1. Derivative of the outside: 2 * (Y - Y_hat)
    /// 2. Derivative of the inside (-Y_hat): -1
    /// 3. Multiply them together: -2 * (Y - Y_hat)
    ///
    /// By distributing the negative sign, we get an optimized, addition-friendly formula:
    /// Formula: dL/dY_hat = 2 * (Y_hat - Y)
    ///
    /// This vector represents the direction of the steepest ASCENT. During backpropagation,
    /// the optimizer will subtract this gradient to move DOWN the error curve.
    pub fn compute_gradient(&self, target: &Array2<f32>, predictions: &Array2<f32>) -> Array2<f32> {
        match self {
            LossFunction::SquaredError => {
                // (Y_hat - Y) * 2.0
                (predictions - target).mapv(|x| x * 2.0)
            }
        }
    }
}
