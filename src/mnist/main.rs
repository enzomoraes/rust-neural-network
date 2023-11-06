mod linear_algebra;
mod linear_algebra_tests;
mod load_mnist_data;
mod neural_network;
use load_mnist_data::load_data;
mod activations;

use neural_network::NeuralNetwork;

use crate::{activations::SIGMOID, linear_algebra::Matrix};

fn main() {
    println!("Hello, world!");
    let activation = SIGMOID;
    let loss_function = &|x: f64| x * x;

    let mut network: NeuralNetwork =
        NeuralNetwork::new(vec![784, 10, 1], 0.3, activation, loss_function);

    let data: load_mnist_data::MNIST = load_data("../data");

    let inputs_train: Vec<Vec<f64>> = data.train_images;
    let target_train: Vec<Vec<f64>> = data.train_labels;
    let inputs_test: Vec<Vec<f64>> = data.test_images;
    let target_test: Vec<Vec<f64>> = data.test_labels;

    network.train(inputs_train, target_train, 1000);
    network.save("./saved-network-mnist.json".to_string());

    let prediction1 = network.try_to_predict(inputs_test[0].clone());
    print!("{} {}", Matrix::new(vec![prediction1]), target_test[0][0]);
}
