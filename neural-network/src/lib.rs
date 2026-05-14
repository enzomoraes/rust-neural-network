pub mod activations;
pub mod layer;
pub mod loss_functions;
pub mod savable_neural_network;

use crate::layer::Layer;
use linear_algebra::Matrix;
use loss_functions::LossFunction;
use std::time::Instant;

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    loss_function: LossFunction,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer>>, loss_function: LossFunction) -> NeuralNetwork {
        return NeuralNetwork {
            layers,
            loss_function,
        };
    }

    pub fn train(&mut self, inputs: Vec<Vec<f32>>, target: Vec<Vec<f32>>, epochs: usize) {
        let start_time = Instant::now();

        for i in 1..=epochs {
            for j in 0..inputs.len() {
                let mut predictions: Vec<f32> = inputs[j].clone();
                for layer_index in 0..self.layers.len() {
                    predictions = self.layers[layer_index].feed_forward(predictions);
                }

                let mut gradient: Matrix = self.loss_function.compute_gradient(
                    &Matrix::from(target[j].clone()),
                    &Matrix::from(predictions.clone()),
                );

                // Backpropagate using the gradient
                for layer_index in (0..self.layers.len()).rev() {
                    gradient = self.layers[layer_index].back_propagate(&gradient);
                }
            }
            // evaluation phase
            if i % 1 == 0 {
                let mut epoch_loss = 0.0;
                let mut correct_predictions = 0;

                for j in 0..inputs.len() {
                    // Feed forward again to see the network's prediction AFTER the epoch's updates
                    let mut predictions: Vec<f32> = inputs[j].clone();
                    for layer_index in 0..self.layers.len() {
                        predictions = self.layers[layer_index].feed_forward(predictions);
                    }

                    // A. Calculate Loss (Mean Squared Error)
                    let loss_matrix = self.loss_function.compute_loss(
                        &Matrix::from(target[j].clone()),
                        &Matrix::from(predictions.clone()),
                    );

                    // Sum the loss for this specific sample
                    let sample_loss: f32 = loss_matrix.data.iter().sum();
                    epoch_loss += sample_loss;

                    // B. Calculate Accuracy
                    let is_correct = if target[j].len() == 1 {
                        // ----------------------------------------------------
                        // LÓGICA BINÁRIA (XOR, AND, OR)
                        // ----------------------------------------------------
                        // Se for maior que 0.5, é 1. Senão, é 0.
                        let predicted_binary = if predictions[0] >= 0.5 { 1.0 } else { 0.0 };
                        let actual_binary = target[j][0];

                        // Compara se o valor previsto bate com o real
                        (predicted_binary - actual_binary).abs() < f32::EPSILON
                    } else {
                        // ----------------------------------------------------
                        // LÓGICA MULTI-CLASSE (MNIST)
                        // ----------------------------------------------------
                        let actual_index = NeuralNetwork::get_max_value_index(target[j].clone());
                        let predicted_index =
                            NeuralNetwork::get_max_value_index(predictions.clone());

                        actual_index == predicted_index
                    };

                    if is_correct {
                        correct_predictions += 1;
                    }
                }

                // Averages
                let mean_loss = epoch_loss / inputs.len() as f32;
                let accuracy = (correct_predictions as f32 / inputs.len() as f32) * 100.0;

                println!(
                    "Epoch {}/{} | Loss: {:.6} | Accuracy: {:.2}% | Elapsed: {} ms",
                    i,
                    epochs,
                    mean_loss,
                    accuracy,
                    start_time.elapsed().as_millis()
                );
            }
        }

        println!(
            "Time elapsed in training -> {:?} miliseconds",
            start_time.elapsed().as_millis()
        );
    }

    pub fn try_to_predict(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let mut predictions: Vec<f32> = inputs;
        for layer_index in 0..self.layers.len() {
            predictions = self.layers[layer_index].feed_forward(predictions);
        }
        return predictions;
    }

    pub fn get_max_value_index(vector: Vec<f32>) -> usize {
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
}
