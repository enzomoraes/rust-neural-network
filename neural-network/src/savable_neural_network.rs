use std::{fs::File, io::{Read, Write}};

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use crate::{NeuralNetwork, layer::{DenseLayer, Layer}, loss_functions::LossFunction};

#[derive(Serialize, Deserialize)]
pub struct SavedNeuralNetwork {
    layers: Vec<SavedLayers>,
    loss_function: LossFunction,
}

#[derive(Serialize, Deserialize)]
pub struct SavedLayers {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
    pub learning_rate: f32,
    pub activation_function: String,
    pub layer_type: String,
    pub inputs: usize,
    pub outputs: usize,
}

pub trait SavableNeuralNetwork {
    fn save(&self, file: String);
    fn load(file: String) -> NeuralNetwork;
    fn load_from_json(json: String) -> NeuralNetwork;
}

impl SavableNeuralNetwork for NeuralNetwork {
    fn save(&self, file: String) {
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

    fn load(file: String) -> NeuralNetwork {
        let mut file = File::open(file).expect("Unable to load saved file");
        let mut buffer = String::new();

        file.read_to_string(&mut buffer)
            .expect("Unable to read saved file");
        let saved_data: SavedNeuralNetwork =
            from_str(&buffer).expect("Unable to serialize saved data");
        let mut layers: Vec<Box<dyn Layer>> = vec![];

        for layer in saved_data.layers.iter() {
            match layer.layer_type.clone().as_str() {
                "dense" => layers.push(Box::new(DenseLayer::from(layer))),
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

    fn load_from_json(json: String) -> NeuralNetwork {
        let saved_data: SavedNeuralNetwork =
            from_str(&json).expect("Unable to serialize saved data");
        let mut layers: Vec<Box<dyn Layer>> = vec![];

        for layer in saved_data.layers.iter() {
            match layer.layer_type.clone().as_str() {
                "dense" => layers.push(Box::new(DenseLayer::from(layer))),
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
