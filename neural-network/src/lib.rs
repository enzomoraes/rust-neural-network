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
        };
    }

    pub fn add_to_weights(&mut self, matrix: &Matrix) {
        self.weights = self.weights.add(&matrix);
    }

    pub fn add_to_biases(&mut self, matrix: &Matrix) {
        self.biases = self.biases.add(&matrix);
    }

    pub fn weigh_inputs(&self, inputs: &Matrix) -> Matrix {
        return self.weights.multiply(&inputs).add(&self.biases);
    }

    fn set_output(&mut self, clone: Matrix) {
        self.output = clone;
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
                let predictions: Vec<f64> = self.feed_forward(inputs[j].clone());
                // precision only works when there are more than 1 output
                if NeuralNetwork::get_max_value_index(target[j].clone())
                    .eq(&NeuralNetwork::get_max_value_index(predictions.clone()))
                {
                    training_precision += 1.0
                }
                self.back_propagate(predictions, target[j].clone(), inputs[j].clone());
            }
            if i % (100) == 0 {
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
        return self.feed_forward(inputs);
    }

    fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0].inputs {
            panic!("Inputs length does not match with neural network");
        }

        let mut data: Matrix = Matrix::new(vec![inputs]).transpose();

        for i in 0..self.layers.len() {
            data = self.layers[i]
                .weigh_inputs(&data)
                .apply_function(&self.activation.function);
            self.layers[i].set_output(data.clone());
        }

        return data.transpose().data[0].to_owned();
    }

    fn back_propagate(&mut self, predictions: Vec<f64>, target: Vec<f64>, inputs: Vec<f64>) {
        if target.len() != self.layers[self.layers.len() - 1].outputs {
            panic!("Invalid targets length");
        }

        let mut loss = self
            .loss(
                &Matrix::new(vec![target]),
                &Matrix::new(vec![predictions.clone()]),
            )
            .transpose();

        let size: usize = self.layers.len();
        let mut gradient: Matrix =
            self.gradient(&Matrix::new(vec![predictions.clone()]).transpose());

        for i in (0..size).rev() {
            gradient = gradient
                .hadamard_product(&loss)
                .apply_function(&|x| x * self.learning_rate);

            if i == 0 {
                let gradient_weights = gradient.multiply(&Matrix::new(vec![inputs.clone()]));
                self.layers[i].add_to_weights(&gradient_weights);
                self.layers[i].add_to_biases(&gradient);
            } else {
                let gradient_weights = gradient.multiply(&self.layers[i - 1].output.transpose());
                self.layers[i].add_to_weights(&gradient_weights);
                self.layers[i].add_to_biases(&gradient);

                loss = self.layers[i]
                    .weights
                    .transpose()
                    // .apply_function(&self.loss_function)
                    .multiply(&loss);
                gradient = self.gradient(&self.layers[i - 1].output);
            }
        }
    }

    fn loss(&self, target: &Matrix, predictions: &Matrix) -> Matrix {
        return target
            // .apply_function(&self.loss_function)
            .subtract(&predictions);
    }

    fn gradient(&self, matrix: &Matrix) -> Matrix {
        return matrix.apply_function(&self.activation.derivative);
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
            })
        }

        self.layers = layers;
    }
}
