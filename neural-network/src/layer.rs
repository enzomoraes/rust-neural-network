use core::fmt;

use crate::{
    activations::ActivationFunction,
    savable_neural_network::SavedLayers,
};
use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub trait Layer {
    fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32>;
    fn back_propagate(&mut self, gradient_from_next_layer: &Array2<f32>) -> Array2<f32>;

    fn get_type(&self) -> String;
    fn get_biases(&self) -> Array2<f32>;
    fn get_weights(&self) -> Array2<f32>;
    fn get_learning_rate(&self) -> f32;
    fn get_activation_function(&self) -> ActivationFunction;
    fn get_inputs(&self) -> usize;
    fn get_outputs(&self) -> usize;
}

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    inputs: usize,
    outputs: usize,
    weights: Array2<f32>,
    biases: Array2<f32>,
    output: Array2<f32>,
    input: Array2<f32>,
    learning_rate: f32,
    activation_function: ActivationFunction,
}

impl Layer for DenseLayer {
    /// Executes the Forward Pass through the Dense Layer.
    ///
    /// The forward pass computes the predictions of this layer given an input vector.
    /// It applies a linear transformation followed by a non-linear activation function.
    ///
    /// # Mathematical Theory
    /// 1. **Linear Transformation ($Z$):** The layer calculates the dot product of its Weights ($W$)
    ///    and the Input vector ($X$), and adds the Biases ($B$).
    ///    $$Z = (W \cdot X) + B$$
    /// 2. **Non-linear Activation ($A$):** The raw weighted sums ($Z$) are passed through the
    ///    configured activation function ($f$) to introduce non-linearity.
    ///    $$A = f(Z)$$
    ///
    /// The resulting activated matrix ($A$) is saved as `self.output` to be reused during
    /// backpropagation, optimizing memory and compute time.
    fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        self.input = Array2::from_shape_vec((inputs.len(), 1), inputs).unwrap();

        // Z = W * X + B
        let weighted_sums = self.weights.dot(&self.input) + &self.biases;

        // A = f(Z)
        let weighted_sums = weighted_sums.mapv(|x| self.activation_function.apply(x));

        self.output = weighted_sums.clone();
        weighted_sums.iter().cloned().collect()
    }

    /// Executes the Backward Pass (Backpropagation) through the Dense Layer.
    ///
    /// This method applies the Chain Rule of Calculus to calculate how much the layer's
    /// weights and biases contributed to the final network error, updates them, and passes
    /// the remaining error backward to the previous layer.
    ///
    /// # Mathematical Theory
    /// 1. **Calculate Local Error ($\delta$):** We multiply the incoming gradient by the derivative
    ///    of the activation function. Because we cached $A$ (`self.output`), we evaluate $f'(A)$.
    ///    The operation $\odot$ represents the Hadamard product (element-wise multiplication).
    ///    $$Error = \text{gradient\_from\_next\_layer} \odot f'(A)$$
    ///
    /// 2. **Calculate Gradients:** We determine how to change weights ($dW$) and biases ($dB$)
    ///    based on the local error and the original inputs ($X$).
    ///    $$dW = Error \cdot X^T$$
    ///    $$dB = Error$$
    ///
    /// 3. **Pass the Baton ($dX$):** We calculate the error gradient to be sent to the previous layer.
    ///    This uses the layer's *current* weights before the update.
    ///    $$dX = W^T \cdot Error$$
    ///
    /// 4. **Gradient Descent Update:** Finally, weights and biases are updated by stepping in the
    ///    opposite direction of the gradient, scaled by the learning rate ($\alpha$).
    ///    $$W_{new} = W - (\alpha \cdot dW)$$
    ///    $$B_{new} = B - (\alpha \cdot dB)$$
    fn back_propagate(&mut self, gradient_from_next_layer: &Array2<f32>) -> Array2<f32> {
        // 1. Error = gradient ⊙ f'(A)
        let error = gradient_from_next_layer * &self.output.mapv(|a| self.activation_function.derivative_a(a));

        // 2. dW = Error * X^T
        let weight_gradients = error.dot(&self.input.t());

        // dB = Error
        let bias_gradients = error.clone();

        // 3. dX = W^T * Error (Calculated before updating weights!)
        let previous_layer_error = self.weights.t().dot(&error);

        // 4. Update Weights and Biases (W = W - α * dW)
        self.add_to_weights(&weight_gradients.mapv(|x| x * -self.learning_rate));
        self.add_to_biases(&bias_gradients.mapv(|x| x * -self.learning_rate));

        previous_layer_error
    }

    fn get_type(&self) -> String {
        String::from("dense")
    }

    fn get_biases(&self) -> Array2<f32> {
        self.biases.clone()
    }

    fn get_weights(&self) -> Array2<f32> {
        self.weights.clone()
    }

    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn get_activation_function(&self) -> ActivationFunction {
        self.activation_function
    }

    fn get_inputs(&self) -> usize {
        self.inputs
    }

    fn get_outputs(&self) -> usize {
        self.outputs
    }
}

impl DenseLayer {
    pub fn new(
        inputs: usize,
        outputs: usize,
        activation_function: ActivationFunction,
        learning_rate: f32,
    ) -> DenseLayer {
        let weights = random_matrix(outputs, inputs, activation_function);
        let biases = Array2::zeros((outputs, 1));

        DenseLayer {
            inputs,
            outputs,
            weights,
            biases,
            output: Array2::zeros((inputs, 1)),
            input: Array2::zeros((outputs, 1)),
            learning_rate,
            activation_function,
        }
    }

    pub fn add_to_weights(&mut self, matrix: &Array2<f32>) {
        self.weights = &self.weights + matrix;
    }

    pub fn add_to_biases(&mut self, matrix: &Array2<f32>) {
        self.biases = &self.biases + matrix;
    }
}

fn random_matrix(rows: usize, cols: usize, activation: ActivationFunction) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let limit = activation.weight_init_limit(cols, rows);

    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-limit..limit))
}

impl From<&SavedLayers> for DenseLayer {
    fn from(layer: &SavedLayers) -> Self {
        DenseLayer {
            inputs: layer.inputs,
            outputs: layer.outputs,
            activation_function: layer.activation_function.clone(),
            biases: layer.biases.clone(),
            weights: layer.weights.clone(),
            input: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            output: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            learning_rate: layer.learning_rate,
        }
    }
}

impl fmt::Display for DenseLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenseLayer: {} -> {} | Activation: {}",
            self.inputs, self.outputs, self.activation_function
        )
    }
}
