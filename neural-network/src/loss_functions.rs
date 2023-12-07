use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum LossFunction {
    SquaredError,
    /// this function has the premisse that target is one-hot encoded
    CategoricalCrossEntropy,
}
