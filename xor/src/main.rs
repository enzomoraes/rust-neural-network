use neural_network::{DenseLayer, NeuralNetwork};

use linear_algebra::Matrix;

fn main() {
    println!("Hello, world!");

    let learning_rate: f32 = 0.3;
    let mut network: NeuralNetwork = NeuralNetwork::new(vec![
        Box::new(DenseLayer::new(2, 3, String::from("TANH"), learning_rate)),
        Box::new(DenseLayer::new(3, 1, String::from("TANH"), learning_rate)),
    ]);
    // let mut network: NeuralNetwork = NeuralNetwork::load("./saved-network.json".to_string());

    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    network.train(inputs, target, 1000);
    network.save("./saved-network.json".to_string());

    let prediction1 = network.try_to_predict(vec![0.0, 0.0]);
    let prediction2 = network.try_to_predict(vec![1.0, 0.0]);
    let prediction3 = network.try_to_predict(vec![0.0, 1.0]);
    let prediction4 = network.try_to_predict(vec![1.0, 1.0]);

    println!("{}", Matrix::from(prediction1));
    println!("{}", Matrix::from(prediction2));
    println!("{}", Matrix::from(prediction3));
    println!("{}", Matrix::from(prediction4));
}
