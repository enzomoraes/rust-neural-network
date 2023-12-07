pub mod activations;
pub mod loss_functions;

use activations::{get_activation_derivative_map, get_activation_map};
use linear_algebra::Matrix;
use loss_functions::LossFunction;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};
use std::{
    fs::File,
    io::{Read, Write},
    time::Instant,
};

pub trait Layer {
    fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32>;
    fn back_propagate(&mut self, loss: &Matrix) -> Matrix;

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
        return weighted_sums.data.clone();
    }
    fn back_propagate(&mut self, loss: &Matrix) -> Matrix {
        let map = get_activation_derivative_map();
        let activation_derivative = map
            .get(&String::from(self.activation_function.clone()))
            .unwrap();
        // Calculate the error for this layer
        let error: Matrix =
            loss.hadamard_product(&self.output.apply_function(activation_derivative));

        // Calculate the gradient of the loss with respect to the weights
        let weight_gradients: Matrix = error.multiply(&self.input.transpose());

        // Calculate the gradient of the loss with respect to the biases
        let bias_gradients: Matrix = error.clone();

        // Update the weights and biases
        self.add_to_weights(&weight_gradients.apply_function(&|x| x * self.learning_rate));
        self.add_to_biases(&bias_gradients.apply_function(&|x| x * self.learning_rate));

        // Calculate the error for the previous layer
        let previous_layer_error: Matrix = self.weights.transpose().multiply(&error);

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
        let weights: Matrix = Matrix::random(outputs, inputs);
        let biases: Matrix = Matrix::random(outputs, 1);

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

#[derive(Serialize, Deserialize)]
pub struct SoftMaxLayer {
    pub inputs: usize,
    pub outputs: usize,
    pub output: Matrix,
}

impl Layer for SoftMaxLayer {
    fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let exp_sum: f32 = inputs.iter().map(|&x| x.exp()).sum();
        let softmax_values: Vec<f32> = inputs.iter().map(|&x| ((x).exp()) / exp_sum).collect();
        self.output = Matrix::from(softmax_values.clone());
        return softmax_values;
    }

    fn back_propagate(&mut self, loss: &Matrix) -> Matrix {
        let target: Matrix = loss.apply_function(&|x| if x.eq(&0f32) { 0f32 } else { 1f32 });
        let gradient = self.output.subtract(&target);
        return gradient;
    }

    fn get_type(&self) -> String {
        return "softmax".to_owned();
    }
    fn get_biases(&self) -> Matrix {
        return Matrix::from(vec![]);
    }
    fn get_weights(&self) -> Matrix {
        return Matrix::from(vec![]);
    }
    fn get_learning_rate(&self) -> f32 {
        return 0.0;
    }
    fn get_activation_function(&self) -> String {
        return "".to_owned();
    }
    fn get_inputs(&self) -> usize {
        return self.inputs;
    }
    fn get_outputs(&self) -> usize {
        return self.outputs;
    }
}

impl SoftMaxLayer {
    pub fn new(inputs: usize, outputs: usize) -> SoftMaxLayer {
        return SoftMaxLayer {
            inputs,
            outputs,
            output: Matrix::zeros(0, 0),
        };
    }
}

#[derive(Serialize, Deserialize)]
pub struct SavedNeuralNetwork {
    layers: Vec<SavedLayers>,
    loss_function: LossFunction,
}

#[derive(Serialize, Deserialize)]
pub struct SavedLayers {
    pub weights: Matrix,
    pub biases: Matrix,
    pub learning_rate: f32,
    pub activation_function: String,
    pub layer_type: String,
    pub inputs: usize,
    pub outputs: usize,
}

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
            let mut training_accuracy: f32 = 0.0;
            for j in 0..inputs.len() {
                let mut predictions: Vec<f32> = inputs[j].clone();
                for layer_index in 0..self.layers.len() {
                    predictions = self.layers[layer_index].feed_forward(predictions);
                }

                let mut loss: Matrix = self.loss_function(
                    &Matrix::from(target[j].clone()),
                    &Matrix::from(predictions.clone()),
                );

                // Backpropagate using the average loss
                for layer_index in (0..self.layers.len()).rev() {
                    loss = self.layers[layer_index].back_propagate(&loss);
                }

                if NeuralNetwork::get_max_value_index(predictions)
                    .eq(&NeuralNetwork::get_max_value_index(target[j].clone()))
                {
                    training_accuracy += 1.0;
                }
            }
            if i % (1) == 0 {
                println!(
                    "Epoch {} of {}. Accuracy: {}% - elapsed {} miliseconds",
                    i,
                    epochs,
                    (training_accuracy / inputs.len() as f32) * 100.0,
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

    fn loss_function(&self, target: &Matrix, predictions: &Matrix) -> Matrix {
        match self.loss_function {
            LossFunction::CategoricalCrossEntropy => {
                return target.hadamard_product(&predictions.apply_function(&|x| -(x.ln())))
            }
            LossFunction::SquaredError => {
                return target
                    .apply_function(&|x| x * x)
                    .subtract(&predictions.apply_function(&|x| x * x));
            }
        }
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

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to create file");
        let mut layers = vec![];
        for layer in &self.layers {
            layers.push(SavedLayers {
                biases: layer.get_biases(),
                weights: layer.get_weights(),
                layer_type: layer.get_type(),
                learning_rate: layer.get_learning_rate(),
                inputs: layer.get_inputs(),
                outputs: layer.get_outputs(),
                activation_function: layer.get_activation_function(),
            })
        }
        file.write_all(
            json!({
              "layers": layers,
              "loss_function": self.loss_function
            })
            .to_string()
            .as_bytes(),
        )
        .expect("Unable to write to file");
    }

    pub fn load(file: String) -> NeuralNetwork {
        let mut file = File::open(file).expect("Unable to load saved file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read saved file");
        let saved_data: SavedNeuralNetwork =
            from_str(&buffer).expect("Unable to serialize saved data");
        let mut layers: Vec<Box<dyn Layer>> = vec![];

        for layer in saved_data.layers.iter() {
            match layer.layer_type.clone().as_str() {
                "dense" => layers.push(Box::new(DenseLayer {
                    inputs: layer.inputs,
                    outputs: layer.outputs,
                    activation_function: layer.activation_function.clone(),
                    biases: layer.biases.clone(),
                    weights: layer.weights.clone(),
                    input: Matrix::new(1, 1, vec![1.0]),
                    output: Matrix::new(1, 1, vec![1.0]),
                    learning_rate: layer.learning_rate,
                })),
                "softmax" => layers.push(Box::new(SoftMaxLayer {
                    inputs: layer.inputs,
                    outputs: layer.outputs,
                    output: Matrix::new(1, 1, vec![1.0]),
                })),
                _ => {
                    panic!("Could not assemble from json")
                }
            }
        }
        return NeuralNetwork {
            layers,
            loss_function: saved_data.loss_function,
        };
    }

    pub fn load_from_json(json: String) -> NeuralNetwork {
        let saved_data: SavedNeuralNetwork =
            from_str(&json).expect("Unable to serialize saved data");
        let mut layers: Vec<Box<dyn Layer>> = vec![];

        for layer in saved_data.layers.iter() {
            match layer.layer_type.clone().as_str() {
                "dense" => layers.push(Box::new(DenseLayer {
                    inputs: layer.inputs,
                    outputs: layer.outputs,
                    activation_function: layer.activation_function.clone(),
                    biases: layer.biases.clone(),
                    weights: layer.weights.clone(),
                    input: Matrix::new(0, 0, vec![]),
                    output: Matrix::new(0, 0, vec![]),
                    learning_rate: layer.learning_rate,
                })),
                "softmax" => layers.push(Box::new(SoftMaxLayer {
                    inputs: layer.inputs,
                    outputs: layer.outputs,
                    output: Matrix::new(0, 0, vec![]),
                })),
                _ => {
                    panic!("Could not assemble from json")
                }
            }
        }
        return NeuralNetwork {
            layers,
            loss_function: saved_data.loss_function,
        };
    }
}
