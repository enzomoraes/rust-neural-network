use std::{collections::HashMap, f32::consts::E};

/// Represents an Activation Function and its corresponding derivatives for a Neural Network.
///
/// In deep learning, an activation function introduces non-linearity to the network,
/// allowing it to learn complex patterns. During the training phase (Backpropagation),
/// the network requires the derivative of this function to calculate the gradient and update weights.
///
/// # The Dual-Derivative Architecture
/// This struct implements a dual-derivative architecture to give the engine choice between
/// mathematical purity and memory/compute optimization:
///
/// * **Pre-activation ($Z$)**: The raw weighted sum calculated by the layer: $Z = (W \cdot X) + b$
/// * **Post-activation ($A$)**: The output of the layer after applying the activation function: $A = f(Z)$
///
/// By providing both `derivative_a` and `derivative_z`, the engine can choose to either cache
/// $Z$ (using more memory but remaining mathematically pure) or reuse the already calculated $A$
/// (saving memory and compute time for specific functions like Sigmoid).
#[derive(Clone)]
pub struct Activation<'a> {
    /// The mathematical function applied during the `feed_forward` pass.
    /// * Formula: $A = f(Z)$
    pub function: &'a dyn Fn(f32) -> f32,

    /// The derivative of the activation function with respect to the post-activation value ($A$).
    /// * Use Case: Highly optimized for memory and compute. Reuses the layer's output (`self.output`).
    /// * Formula: $f'(A)$
    pub derivative_a: &'a dyn Fn(f32) -> f32,

    /// The derivative of the activation function with respect to the pre-activation value ($Z$).
    /// * Use Case: Mathematically exact. Requires the layer to cache the raw weighted sums during feed forward.
    /// * Formula: $f'(Z)$
    pub derivative_z: &'a dyn Fn(f32) -> f32,
}

/// # Identity (Linear) Activation
/// Used primarily for regression tasks in the output layer.
/// It passes the value through without any modifications.
///
/// * **Function:** $f(Z) = Z$
/// * **Derivative:** $f'(Z) = 1$ (The rate of change is constant)
pub const IDENTITY: Activation = Activation {
    function: &|x| x,
    derivative_a: &|_| 1.0,
    derivative_z: &|_| 1.0,
};

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
pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative_a: &|a| a * (1.0 - a),
    derivative_z: &|z| {
        let s = 1.0 / (1.0 + E.powf(-z));
        s * (1.0 - s)
    },
};

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
pub const TANH: Activation = Activation {
    function: &|x| x.tanh(),
    derivative_a: &|a| 1.0 - (a.powi(2)),
    derivative_z: &|z| {
        let t = z.tanh();
        1.0 - (t.powi(2))
    },
};

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
pub const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative_a: &|a| if a > 0.0 { 1.0 } else { 0.0 },
    derivative_z: &|z| if z > 0.0 { 1.0 } else { 0.0 },
};

// ==========================================
// Map Generators
// ==========================================

/// Returns a map of standard Activation Functions for the `feed_forward` pass.
pub fn get_activation_map() -> HashMap<String, &'static dyn Fn(f32) -> f32> {
    let mut activation_map: HashMap<String, &'static dyn Fn(f32) -> f32> = HashMap::new();
    activation_map.insert(String::from("IDENTITY"), IDENTITY.function);
    activation_map.insert(String::from("SIGMOID"), SIGMOID.function);
    activation_map.insert(String::from("TANH"), TANH.function);
    activation_map.insert(String::from("RELU"), RELU.function);
    activation_map
}

/// Returns a map of derivatives optimized for Post-activation values ($A$).
/// Requires passing `layer.output` during backpropagation.
pub fn get_activation_derivative_a_map() -> HashMap<String, &'static dyn Fn(f32) -> f32> {
    let mut activation_map: HashMap<String, &'static dyn Fn(f32) -> f32> = HashMap::new();
    activation_map.insert(String::from("IDENTITY"), IDENTITY.derivative_a);
    activation_map.insert(String::from("SIGMOID"), SIGMOID.derivative_a);
    activation_map.insert(String::from("TANH"), TANH.derivative_a);
    activation_map.insert(String::from("RELU"), RELU.derivative_a);
    activation_map
}

/// Returns a map of pure mathematical derivatives for Pre-activation values ($Z$).
/// Requires the layer to cache the `Z` matrix (weighted sums) during the forward pass.
pub fn get_activation_derivative_z_map() -> HashMap<String, &'static dyn Fn(f32) -> f32> {
    let mut activation_map: HashMap<String, &'static dyn Fn(f32) -> f32> = HashMap::new();
    activation_map.insert(String::from("IDENTITY"), IDENTITY.derivative_z);
    activation_map.insert(String::from("SIGMOID"), SIGMOID.derivative_z);
    activation_map.insert(String::from("TANH"), TANH.derivative_z);
    activation_map.insert(String::from("RELU"), RELU.derivative_z);
    activation_map
}
