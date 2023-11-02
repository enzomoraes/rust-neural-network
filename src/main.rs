mod linear_algebra;
mod linear_algebra_tests;
mod neural_network;

use neural_network::NeuralNetwork;

use crate::neural_network::Activation;

fn main() {
    println!("Hello, world!");
    let activation: Activation = Activation {
        function: Box::new(&|x: f64| x.max(0.0)),
        derivative: Box::new(&|_x| 1.0),
    };
    let loss_function = Box::new(&|x: f64| x.max(0.0));

    let learning_rate = 0.5;
    let network: NeuralNetwork =
        NeuralNetwork::new(vec![2, 3, 2], learning_rate, activation, loss_function);

    println!("{}", network.feed_forward(&vec![0.0, 0.0]));
}
