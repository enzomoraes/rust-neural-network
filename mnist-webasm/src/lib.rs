use neural_network::NeuralNetwork;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn try_to_predict(pixels: Vec<f32>, json: String) -> Vec<f32> {
    let mut network: NeuralNetwork = NeuralNetwork::load_from_json(json.to_string());

    return network.try_to_predict(pixels);
}
