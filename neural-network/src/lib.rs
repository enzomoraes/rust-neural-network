pub mod activations;

use activations::Activation;
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};
use std::{
    fs::File,
    io::{Read, Write},
    time::Instant,
};

#[derive(Serialize, Deserialize)]
pub struct Layer<'a> {
    inputs: usize,
    outputs: usize,
    learning_rate: f32,

    #[serde(skip)]
    activation: Activation<'a>,
    weights: Array2<f32>,
    biases: Array2<f32>,
    output: Array2<f32>,
    input: Array2<f32>,
}

impl<'a> Layer<'a> {
    pub fn new(
        inputs: usize,
        outputs: usize,
        learning_rate: f32,
        activation: Activation<'a>,
    ) -> Layer<'a> {
        let weights: Array2<f32> = Array2::random((outputs, inputs), Uniform::new(-1.0, 1.0));
        let biases: Array2<f32> = Array2::random((outputs, 1), Uniform::new(-1.0, 1.0));

        return Layer {
            inputs,
            outputs,
            learning_rate,
            weights,
            biases,
            activation,
            output: Array2::zeros((inputs, 1)),
            input: Array2::zeros((outputs, 1)),
        };
    }

    pub fn feed_forward(&mut self, inputs: Array2<f32>) -> Array2<f32> {
        // Convert the input vector to a matrix
        self.input = inputs.clone();
        // Calculate the weighted sum of the inputs
        let weighted_sums: Array2<f32> = (self.weights.dot(&self.input) + &self.biases)
            .map(|x: &f32| (self.activation.function)(*x));
        self.output = weighted_sums.clone();
        // Return the output vector
        return weighted_sums.clone();
    }

    pub fn back_propagate(&mut self, loss: &Array2<f32>) -> Array2<f32> {
        // Calculate the error for this layer
        let error: Array2<f32> =
            loss * &self.output.map(|x: &f32| (self.activation.derivative)(*x));
        // Calculate the gradient of the loss with respect to the weights
        let weight_gradients: Array2<f32> = error.dot(&self.input.t());

        // Calculate the gradient of the loss with respect to the biases
        let bias_gradients: Array2<f32> = error.clone();

        // Update the weights and biases
        self.add_to_weights(&weight_gradients.map(&|x: &f32| x * self.learning_rate));
        self.add_to_biases(&bias_gradients.map(&|x: &f32| x * self.learning_rate));

        // Calculate the error for the previous layer
        let previous_layer_error: Array2<f32> = self.weights.t().dot(&error);

        // Return the error for the previous layer
        return previous_layer_error;
    }

    pub fn add_to_weights(&mut self, matrix: &Array2<f32>) {
        self.weights = self.weights.clone() + matrix;
    }

    pub fn add_to_biases(&mut self, matrix: &Array2<f32>) {
        self.biases = self.biases.clone() + matrix;
    }
}

pub struct NeuralNetwork<'a> {
    layers: Vec<Layer<'a>>,
    learning_rate: f32,
    activation: Activation<'a>,
}

#[derive(Serialize, Deserialize)]
struct SavedNeuralNetwork<'a> {
    layers: Vec<Layer<'a>>,
}

impl NeuralNetwork<'_> {
    pub fn new<'a>(
        neurons_per_layer: Vec<usize>,
        learning_rate: f32,
        activation: Activation<'a>,
    ) -> NeuralNetwork<'a> {
        let mut layers: Vec<Layer<'a>> = vec![];

        for i in 1..neurons_per_layer.len() {
            layers.push(Layer::new(
                neurons_per_layer[i - 1],
                neurons_per_layer[i],
                learning_rate,
                activation.clone(),
            ));
        }

        return NeuralNetwork {
            layers,
            learning_rate,
            activation,
        };
    }

    pub fn train(&mut self, inputs: Vec<Vec<f32>>, target: Vec<Vec<f32>>, epochs: usize) {
        let start_time = Instant::now();

        for i in 1..=epochs {
            let mut training_precision: f32 = 0.0;
            for j in 0..inputs.len() {
                let mut predictions: Array2<f32> =
                    Array2::from_shape_vec((inputs[j].len(), 1), inputs[j].clone()).unwrap();
                for layer_index in 0..self.layers.len() {
                    predictions = self.layers[layer_index].feed_forward(predictions);
                }
                let mut loss: Array2<f32> = self.loss_derivative(
                    &Array2::from_shape_vec((target[j].len(), 1), target[j].clone()).unwrap(),
                    &predictions,
                );
                // Backpropagate using the average loss
                for layer_index in (0..self.layers.len()).rev() {
                    loss = self.layers[layer_index].back_propagate(&loss);
                }

                if NeuralNetwork::find_max_index_and_value(&predictions).eq(
                    &NeuralNetwork::find_max_index_and_value(
                        &Array2::from_shape_vec((target[j].len(), 1), target[j].clone()).unwrap(),
                    ),
                ) {
                    training_precision += 1.0;
                }
            }
            if i % (1) == 0 {
                println!(
                    "Epoch {} of {}. Precision: {}%",
                    i,
                    epochs,
                    (training_precision / inputs.len() as f32) * 100.0
                );
            }
        }
        println!(
            "Time elapsed in training -> {:?} miliseconds",
            start_time.elapsed().as_millis()
        );
    }

    pub fn try_to_predict(&mut self, inputs: Array2<f32>) -> Array2<f32> {
        let mut predictions: Array2<f32> = inputs;
        for layer_index in 0..self.layers.len() {
            predictions = self.layers[layer_index].feed_forward(predictions);
        }
        return predictions;
    }

    fn loss_derivative(&self, target: &Array2<f32>, predictions: &Array2<f32>) -> Array2<f32> {
        return (target - predictions).map(&|x| 2.0 * x / target.row(0).len() as f32);
    }

    pub fn find_max_index_and_value(arr: &Array2<f32>) -> usize {
        return arr
            .axis_iter(Axis(0))
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |acc, (idx, vec)| {
                let max_value = vec[0].max(acc.1);
                (if max_value > acc.1 { idx } else { acc.0 }, max_value)
            })
            .0;
    }

    pub fn save(&self, file: String) {
        let mut file = File::create(file).expect("Unable to create file");

        file.write_all(
            json!({
              "layers": self.layers
            })
            .to_string()
            .as_bytes(),
        )
        .expect("Unable to write to file");
    }

    pub fn load(&mut self, file: String) {
        let mut file = File::open(file).expect("Unable to load saved file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read saved file");

        let saved_data: SavedNeuralNetwork =
            from_str(&buffer).expect("Unable to serialize saved data");

        let mut layers: Vec<Layer> = vec![];

        for i in 0..self.layers.len() {
            layers.push(Layer {
                inputs: self.layers[i].inputs,
                outputs: self.layers[i].outputs,
                learning_rate: self.learning_rate,
                activation: self.activation.clone(),
                weights: Array2::from(saved_data.layers[i].weights.clone()),
                biases: Array2::from(saved_data.layers[i].biases.clone()),
                output: saved_data.layers[i].output.clone(),
                input: saved_data.layers[i].input.clone(),
            })
        }

        self.layers = layers;
    }
}
