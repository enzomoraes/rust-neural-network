use std::f32::consts::E;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Activation functions for neural networks.
///
/// Each variant represents a different activation function used in neural network layers.
/// Activation functions introduce non-linearity to the network, allowing it to learn complex patterns.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// # Identity (Linear) Activation
    /// Used primarily for regression tasks in the output layer.
    /// It passes the value through without any modifications.
    ///
    /// * **Function:** $f(Z) = Z$
    /// * **Derivative:** $f'(Z) = 1$ (The rate of change is constant)
    Identity,

    /// # Sigmoid Activation
    /// Squeezes numbers into a range between `0.0` and `1.0`. Excellent for probability and binary classification.
    ///
    /// * **Function:** $f(Z) = \frac{1}{1 + e^{-Z}}$
    ///
    /// ### The Optimization Theory:
    /// The raw mathematical derivative is $f'(Z) = f(Z) \cdot (1 - f(Z))$.
    /// Because recalculating $e^{-Z}$ is computationally expensive, and we know that $A = f(Z)$,
    /// we can dramatically optimize the derivative by substituting $A$ into the formula:
    /// * **Derivative via $A$:** $A \cdot (1 - A)$
    Sigmoid,

    /// # Hyperbolic Tangent (Tanh) Activation
    /// Similar to Sigmoid, but squashes values between `-1.0` and `1.0`. Generally preferred over
    /// Sigmoid in hidden layers because its output is zero-centered.
    ///
    /// * **Function:** $f(Z) = \tanh(Z)$
    ///
    /// ### The Optimization Theory:
    /// The standard derivative is $f'(Z) = 1 - \tanh^2(Z)$.
    /// Since we already computed $A = \tanh(Z)$ during the forward pass, we can skip the expensive
    /// trigonometric recalculation by substituting $A$:
    /// * **Derivative via $A$:** $1 - A^2$
    Tanh,

    /// # Rectified Linear Unit (ReLU) Activation
    /// It solves the vanishing gradient problem by keeping a constant derivative for positive numbers.
    ///
    /// * **Function:** $f(Z) = \max(0, Z)$
    ///
    /// ### The Optimization Theory:
    /// The derivative is simply `1.0` if the input is positive, and `0.0` if it is negative.
    /// Because ReLU maps all negative numbers to `0` and leaves positive numbers unchanged,
    /// testing if $A > 0$ yields the exact same logical result as testing if $Z > 0$.
    /// Therefore, `derivative_a` and `derivative_z` share the exact same logic.
    /// * **Derivative:** `1.0` if $x > 0$ else `0.0`
    Relu,
}

impl fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ActivationFunction::Identity => "Identity",
                ActivationFunction::Sigmoid => "Sigmoid",
                ActivationFunction::Tanh => "Tanh",
                ActivationFunction::Relu => "Relu",
            }
        )
    }
}

impl ActivationFunction {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Identity => x,
            ActivationFunction::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Relu => x.max(0.0),
        }
    }

    pub fn derivative_a(&self, a: f32) -> f32 {
        match self {
            ActivationFunction::Identity => 1.0,
            ActivationFunction::Sigmoid => a * (1.0 - a),
            ActivationFunction::Tanh => 1.0 - (a.powi(2)),
            ActivationFunction::Relu => if a > 0.0 { 1.0 } else { 0.0 },
        }
    }

    pub fn derivative_z(&self, z: f32) -> f32 {
        match self {
            ActivationFunction::Identity => 1.0,
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + E.powf(-z));
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => {
                let t = z.tanh();
                1.0 - (t.powi(2))
            }
            ActivationFunction::Relu => if z > 0.0 { 1.0 } else { 0.0 },
        }
    }

    pub fn weight_init_limit(&self, input_size: usize, output_size: usize) -> f32 {
        match self {
            ActivationFunction::Relu => (6.0 / (input_size as f32)).sqrt(),
            ActivationFunction::Sigmoid | ActivationFunction::Tanh => {
                (6.0 / (input_size + output_size) as f32).sqrt()
            }
            ActivationFunction::Identity => 1.0 / (input_size as f32).sqrt(),
        }
    }
}
