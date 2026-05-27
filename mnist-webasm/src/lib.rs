use neural_network::NeuralNetwork;
use neural_network::savable_neural_network::SavableNeuralNetwork;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn try_to_predict(pixels: Vec<f32>, json: String) -> Vec<f32> {
    let mut network: NeuralNetwork = NeuralNetwork::load_from_json(json.to_string());

    return network.try_to_predict(pixels);
}

#[wasm_bindgen]
pub fn try_to_predict_with_trace(pixels: Vec<f32>, json: String) -> Result<JsValue, JsError> {
    if pixels.is_empty() {
        return Err(JsError::new("pixels must not be empty"));
    }
    let mut network: NeuralNetwork = NeuralNetwork::load_from_json(json);
    let prediction = network.try_to_predict_with_trace(pixels);
    serde_wasm_bindgen::to_value(&prediction).map_err(|e| JsError::new(&e.to_string()))
}
