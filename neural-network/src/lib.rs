pub mod activations;
pub mod layer;
pub mod loss_functions;
pub mod savable_neural_network;

use crate::layer::Layer;
use loss_functions::LossFunction;
use ndarray::Array2;
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

    fn evaluate_epoch(&mut self, inputs: &Vec<Vec<f32>>, target: &Vec<Vec<f32>>) -> (f32, f32) {
        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;

        for j in 0..inputs.len() {
            let mut predictions = Array2::from_shape_vec((inputs[j].len(), 1), inputs[j].clone()).unwrap();
            for layer_index in 0..self.layers.len() {
                predictions = self.layers[layer_index].feed_forward(&predictions);
            }

            let target_array = Array2::from_shape_vec((target[j].len(), 1), target[j].clone()).unwrap();
            let loss_matrix = self.loss_function.compute_loss(&target_array, &predictions);
            let sample_loss: f32 = loss_matrix.iter().sum();
            epoch_loss += sample_loss;

            let predictions_vec: Vec<f32> = predictions.iter().cloned().collect();
            let is_correct = if target[j].len() == 1 {
                let predicted_binary = if predictions_vec[0] >= 0.5 { 1.0 } else { 0.0 };
                let actual_binary = target[j][0];
                (predicted_binary - actual_binary).abs() < f32::EPSILON
            } else {
                let actual_index = NeuralNetwork::get_max_value_index(target[j].clone());
                let predicted_index = NeuralNetwork::get_max_value_index(predictions_vec);
                actual_index == predicted_index
            };

            if is_correct {
                correct_predictions += 1;
            }
        }

        let mean_loss = epoch_loss / inputs.len() as f32;
        let accuracy = (correct_predictions as f32 / inputs.len() as f32) * 100.0;

        (mean_loss, accuracy)
    }

    pub fn train(&mut self, inputs: Vec<Vec<f32>>, target: Vec<Vec<f32>>, epochs: usize) {
        let start_time = Instant::now();

        for i in 1..=epochs {
            for j in 0..inputs.len() {
                let mut predictions = Array2::from_shape_vec((inputs[j].len(), 1), inputs[j].clone()).unwrap();
                for layer_index in 0..self.layers.len() {
                    predictions = self.layers[layer_index].feed_forward(&predictions);
                }

                let target_array = Array2::from_shape_vec((target[j].len(), 1), target[j].clone()).unwrap();
                let mut gradient = self.loss_function.compute_gradient(&target_array, &predictions);

                // Backpropagate using the gradient
                for layer_index in (0..self.layers.len()).rev() {
                    gradient = self.layers[layer_index].back_propagate(&gradient);
                }
            }
            // evaluation phase
            if i % 1 == 0 {
                let (mean_loss, accuracy) = self.evaluate_epoch(&inputs, &target);
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

    pub fn train_with_batches(&mut self, inputs: Vec<Vec<f32>>, target: Vec<Vec<f32>>, epochs: usize, batch_size: usize) {
        let start_time = Instant::now();

        for epoch in 1..=epochs {
            for batch_start in (0..inputs.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(inputs.len());
                let actual_batch_size = batch_end - batch_start;

                let mut accumulated_weight_grads: Vec<Option<Array2<f32>>> = vec![None; self.layers.len()];
                let mut accumulated_bias_grads: Vec<Option<Array2<f32>>> = vec![None; self.layers.len()];

                for sample_idx in batch_start..batch_end {
                    let mut predictions = Array2::from_shape_vec((inputs[sample_idx].len(), 1), inputs[sample_idx].clone()).unwrap();
                    for layer in &mut self.layers {
                        predictions = layer.feed_forward(&predictions);
                    }

                    let target_array = Array2::from_shape_vec((target[sample_idx].len(), 1), target[sample_idx].clone()).unwrap();
                    let mut gradient = self.loss_function.compute_gradient(&target_array, &predictions);

                    for (layer_idx, layer) in self.layers.iter_mut().enumerate().rev() {
                        let (prev_gradient, w_grad, b_grad) = layer.compute_gradients(&gradient);

                        if accumulated_weight_grads[layer_idx].is_none() {
                            accumulated_weight_grads[layer_idx] = Some(w_grad);
                            accumulated_bias_grads[layer_idx] = Some(b_grad);
                        } else {
                            accumulated_weight_grads[layer_idx] = Some(accumulated_weight_grads[layer_idx].take().unwrap() + &w_grad);
                            accumulated_bias_grads[layer_idx] = Some(accumulated_bias_grads[layer_idx].take().unwrap() + &b_grad);
                        }

                        gradient = prev_gradient;
                    }
                }

                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    let w_grads = accumulated_weight_grads[layer_idx].as_ref().unwrap();
                    let b_grads = accumulated_bias_grads[layer_idx].as_ref().unwrap();
                    layer.apply_weight_update(w_grads, b_grads, actual_batch_size);
                }
            }

            if epoch % 1 == 0 {
                let (mean_loss, accuracy) = self.evaluate_epoch(&inputs, &target);
                println!(
                    "Epoch {}/{} | Loss: {:.6} | Accuracy: {:.2}% | Elapsed: {} ms",
                    epoch, epochs, mean_loss, accuracy, start_time.elapsed().as_millis()
                );
            }
        }

        println!("Time elapsed in training -> {:?} milliseconds", start_time.elapsed().as_millis());
    }

    pub fn try_to_predict(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let mut predictions = Array2::from_shape_vec((inputs.len(), 1), inputs).unwrap();
        for layer_index in 0..self.layers.len() {
            predictions = self.layers[layer_index].feed_forward(&predictions);
        }
        predictions.iter().cloned().collect()
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
