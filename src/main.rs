mod linear_algebra;
mod linear_algebra_tests;
mod neural_network;

use std::f64::consts::E;

use neural_network::NeuralNetwork;

use crate::{linear_algebra::Matrix, neural_network::Activation};

fn main() {
    println!("Hello, world!");
    let activation: Activation = Activation {
        function: Box::new(&|x: f64| 1.0 / (1.0 + E.powf(-x))),
        derivative: Box::new(&|x: f64| x * (1.0 - x)),
    };
    let loss_function = Box::new(&|x: f64| x * x);

    let learning_rate = 0.2;
    let mut network: NeuralNetwork =
        NeuralNetwork::new(vec![2, 3, 1], learning_rate, activation, loss_function);

    let inputs: &Vec<Vec<f64>> = &vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target: &Vec<Vec<f64>> = &vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    network.train(&inputs, target, 1000);

    let prediction1 = network.try_to_predict(&inputs[0]);
    let prediction2 = network.try_to_predict(&inputs[1]);
    let prediction3 = network.try_to_predict(&inputs[2]);
    let prediction4 = network.try_to_predict(&inputs[3]);

    println!("{}", Matrix::new(vec![prediction1]));
    println!("{}", Matrix::new(vec![prediction2]));
    println!("{}", Matrix::new(vec![prediction3]));
    println!("{}", Matrix::new(vec![prediction4]));
}
