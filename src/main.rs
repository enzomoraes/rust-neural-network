mod linear_algebra;
mod linear_algebra_tests;
mod network;

use linear_algebra::Matrix;
use network::NeuralNetwork;

fn main() {
    println!("Hello, world!");
    let inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let activation = Box::new(&|x: f64| x.max(0.0));
    let learning_rate = 0.5;
    let network: NeuralNetwork = NeuralNetwork::new(vec![2, 3, 1], learning_rate, activation);

    println!("{}", network.feed_forward(vec![0.0, 0.0]));
    println!("{}", network.feed_forward(vec![0.0, 1.0]));
    println!("{}", network.feed_forward(vec![1.0, 0.0]));
    println!("{}", network.feed_forward(vec![1.0, 1.0]));
}
