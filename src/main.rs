mod linear_algebra;
mod linear_algebra_tests;
mod neural_network;

use neural_network::NeuralNetwork;

fn main() {
    println!("Hello, world!");
    let activation = Box::new(&|x: f64| x.max(0.0));
    let learning_rate = 0.5;
    let network: NeuralNetwork = NeuralNetwork::new(vec![2, 3, 2], learning_rate, activation);

    println!("{}", network.feed_forward(&vec![0.0, 0.0]));
}
