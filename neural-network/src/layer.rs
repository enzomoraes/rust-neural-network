use core::fmt;

use crate::{activations::{get_activation_derivative_map, get_activation_map}, savable_neural_network::SavedLayers};
use linear_algebra::Matrix;
use serde::{Deserialize, Serialize};

pub trait Layer {
    fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32>;
    fn back_propagate(&mut self, gradient_from_next_layer: &Matrix) -> Matrix;

    fn get_type(&self) -> String;
    fn get_biases(&self) -> Matrix;
    fn get_weights(&self) -> Matrix;
    fn get_learning_rate(&self) -> f32;
    fn get_activation_function(&self) -> String;
    fn get_inputs(&self) -> usize;
    fn get_outputs(&self) -> usize;
}

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    inputs: usize,
    outputs: usize,
    weights: Matrix,
    biases: Matrix,
    output: Matrix,
    input: Matrix,
    learning_rate: f32,
    activation_function: String,
}

impl Layer for DenseLayer {
    fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        // Convert the input vector to a matrix
        self.input = Matrix::from(inputs);
        let map = get_activation_map();
        let activation_function = map
            .get(&String::from(self.activation_function.clone()))
            .unwrap();
        // Calculate the weighted sum of the inputs
        let weighted_sums: Matrix = self
            .weights
            .multiply(&self.input)
            .add(&self.biases)
            .apply_function(&activation_function);

        self.output = weighted_sums.clone();
        // Return the output vector
        return weighted_sums.data;
    }
    fn back_propagate(&mut self, gradient_from_next_layer: &Matrix) -> Matrix {
        let map = get_activation_derivative_map();
        let activation_derivative = map
            .get(&String::from(self.activation_function.clone()))
            .unwrap();
        // Calculate the error for this layer
        let error: Matrix = gradient_from_next_layer
            .hadamard_product(&self.output.apply_function(activation_derivative));

        // Calculate the gradient of the loss with respect to the weights
        let weight_gradients: Matrix = error.multiply(&self.input.transpose());

        // Calculate the gradient of the loss with respect to the biases
        let bias_gradients: Matrix = error.clone();

        // Calculate the error for the previous layer
        let previous_layer_error: Matrix = self.weights.transpose().multiply(&error);

        // Update the weights and biases
        self.add_to_weights(&weight_gradients.apply_function(&|x| x * -self.learning_rate));
        self.add_to_biases(&bias_gradients.apply_function(&|x| x * -self.learning_rate));

        // Return the error for the previous layer
        return previous_layer_error;
    }

    fn get_type(&self) -> String {
        return String::from("dense");
    }
    fn get_biases(&self) -> Matrix {
        return self.biases.clone();
    }
    fn get_weights(&self) -> Matrix {
        return self.weights.clone();
    }
    fn get_learning_rate(&self) -> f32 {
        return self.learning_rate;
    }
    fn get_activation_function(&self) -> String {
        return self.activation_function.clone();
    }
    fn get_inputs(&self) -> usize {
        return self.inputs;
    }
    fn get_outputs(&self) -> usize {
        return self.outputs;
    }
}

impl DenseLayer {
    pub fn new(
        inputs: usize,
        outputs: usize,
        activation_function: String,
        learning_rate: f32,
    ) -> DenseLayer {
      
        let weights: Matrix = Matrix::random(outputs, inputs, &activation_function);
        let biases: Matrix = Matrix::zeros(outputs, 1);

        return DenseLayer {
            inputs,
            outputs,
            weights,
            biases,
            output: Matrix::zeros(inputs, 1),
            input: Matrix::zeros(outputs, 1),
            learning_rate,
            activation_function,
        };
    }

    pub fn add_to_weights(&mut self, matrix: &Matrix) {
        self.weights = self.weights.add(&matrix);
    }

    pub fn add_to_biases(&mut self, matrix: &Matrix) {
        self.biases = self.biases.add(&matrix);
    }
}

impl From<&SavedLayers> for DenseLayer {
    fn from(layer: &SavedLayers) -> Self {
        DenseLayer {
            inputs: layer.inputs,
            outputs: layer.outputs,
            activation_function: layer.activation_function.clone(),
            biases: layer.biases.clone(),
            weights: layer.weights.clone(),
            input: Matrix::new(1, 1, vec![1.0]),
            output: Matrix::new(1, 1, vec![1.0]),
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