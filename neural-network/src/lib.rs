pub mod activations;

use activations::Activation;
use linear_algebra::Matrix;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};
use std::{
    fs::File,
    io::{Read, Write},
};

#[derive(Serialize, Deserialize)]
pub struct Layer {
    inputs: usize,
    outputs: usize,
    weights: Matrix,
    biases: Matrix,
    output: Matrix,
    input: Matrix,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let weights: Matrix = Matrix::random(outputs, inputs);
        let biases: Matrix = Matrix::random(outputs, 1);

        return Layer {
            inputs,
            outputs,
            weights,
            biases,
            output: Matrix::zero(inputs, 1),
            input: Matrix::zero(outputs, 1),
        };
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        // Convert the input vector to a matrix
        self.input = Matrix::new(vec![inputs]).transpose();

        // Calculate the weighted sum of the inputs
        let weighted_sums: Matrix = self
            .weights
            .multiply(&self.input)
            .add(&self.biases)
            .apply_function(activations::SIGMOID.function);

        self.output = weighted_sums.clone();
        // Return the output vector
        return weighted_sums.transpose().data[0].clone();
    }

    pub fn back_propagate(&mut self, loss: &Matrix) -> Matrix {
        // Calculate the error for this layer
        let error: Matrix =
            loss.hadamard_product(&self.output.apply_function(activations::SIGMOID.derivative));

        // Calculate the gradient of the loss with respect to the weights
        let weight_gradients: Matrix = error.multiply(&self.input.transpose());

        // Calculate the gradient of the loss with respect to the biases
        let bias_gradients: Matrix = error.clone();

        // Update the weights and biases
        self.add_to_weights(&weight_gradients.apply_function(&|x| x * 0.3));
        self.add_to_biases(&bias_gradients.apply_function(&|x| x * 0.3));

        // Calculate the error for the previous layer
        let previous_layer_error: Matrix = self.weights.transpose().multiply(&error);

        // Return the error for the previous layer
        return previous_layer_error;
    }

    pub fn add_to_weights(&mut self, matrix: &Matrix) {
        self.weights = self.weights.add(&matrix);
    }

    pub fn add_to_biases(&mut self, matrix: &Matrix) {
        self.biases = self.biases.add(&matrix);
    }
}

pub struct NeuralNetwork<'a> {
    layers: Vec<Layer>,
    learning_rate: f64,
    activation: Activation<'a>,
    loss_function: &'a dyn Fn(f64) -> f64,
}

#[derive(Serialize, Deserialize)]
struct SavedNeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork<'_> {
    pub fn new<'a>(
        neurons_per_layer: Vec<usize>,
        learning_rate: f64,
        activation: Activation<'a>,
        loss_function: &'a dyn Fn(f64) -> f64,
    ) -> NeuralNetwork<'a> {
        let mut layers: Vec<Layer> = vec![];

        for i in 1..neurons_per_layer.len() {
            layers.push(Layer::new(neurons_per_layer[i - 1], neurons_per_layer[i]));
        }

        return NeuralNetwork {
            layers,
            learning_rate,
            activation,
            loss_function,
        };
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, target: Vec<Vec<f64>>, epochs: usize) {
        for i in 1..=epochs {
            let mut training_precision: f64 = 0.0;
            for j in 0..inputs.len() {
                let mut predictions: Vec<f64> = inputs[j].clone();
                for layer_index in 0..self.layers.len() {
                    predictions = self.layers[layer_index].feed_forward(predictions);
                }
                let mut loss: Matrix = self
                    .loss_derivative(
                        &Matrix::new(vec![target[j].clone()]),
                        &Matrix::new(vec![predictions.clone()]),
                    )
                    .transpose();

                // Backpropagate using the average loss
                for layer_index in (0..self.layers.len()).rev() {
                    loss = self.layers[layer_index].back_propagate(&loss);
                }

                if NeuralNetwork::get_max_value_index(predictions)
                    .eq(&NeuralNetwork::get_max_value_index(target[j].clone()))
                {
                    training_precision += 1.0;
                }
            }
            if i % (1) == 0 {
                println!(
                    "Epoch {} of {}. Precision: {}%",
                    i,
                    epochs,
                    (training_precision / inputs.len() as f64) * 100.0
                );
            }
        }
    }

    pub fn try_to_predict(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut predictions: Vec<f64> = inputs;
        for layer_index in 0..self.layers.len() {
            predictions = self.layers[layer_index].feed_forward(predictions);
        }
        return predictions;
    }

    fn loss_derivative(&self, target: &Matrix, predictions: &Matrix) -> Matrix {
        return target
            .subtract(&predictions)
            .apply_function(&|x| 2.0 * x / target.rows as f64);
    }

    pub fn get_max_value_index(vector: Vec<f64>) -> usize {
        let mut max_index = 0;
        let mut max_value = vector[0];

        for (index, &value) in vector.iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = index;
            }
        }
        return max_index;
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to touch save file");

        file.write_all(
            json!({
              "layers": self.layers
            })
            .to_string()
            .as_bytes(),
        )
        .expect("Unable to write to save file");
    }

    pub fn load(&mut self, file: String) {
        let mut file = File::open(file).expect("Unable to open save file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read save file");

        let saved_data: SavedNeuralNetwork =
            from_str(&buffer).expect("Unable to serialize save data");

        let mut layers: Vec<Layer> = vec![];

        for i in 0..self.layers.len() {
            layers.push(Layer {
                inputs: self.layers[i].inputs,
                outputs: self.layers[i].outputs,
                weights: Matrix::from(saved_data.layers[i].weights.clone()),
                biases: Matrix::from(saved_data.layers[i].biases.clone()),
                output: saved_data.layers[i].output.clone(),
                input: saved_data.layers[i].input.clone(),
            })
        }

        self.layers = layers;
    }
}
