use neural_network::activations::SIGMOID;
use neural_network::NeuralNetwork;

use linear_algebra::Matrix;

fn main() {
    println!("Hello, world!");
    let activation = SIGMOID;
    let loss_function = &|x: f64| x * x;

    // let mut network: NeuralNetwork =
    //     NeuralNetwork::new(vec![2, 3, 1], 0.2, activation, loss_function);
    let mut network = NeuralNetwork::new(vec![2, 3, 1], 0.2, activation, loss_function);
    network.load("./saved-network.json".to_string());

    let inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    network.train(inputs, target, 10000);
    network.save("./saved-network.json".to_string());

    let prediction1 = network.try_to_predict(vec![0.0, 0.0]);
    let prediction2 = network.try_to_predict(vec![1.0, 0.0]);
    let prediction3 = network.try_to_predict(vec![0.0, 1.0]);
    let prediction4 = network.try_to_predict(vec![1.0, 1.0]);

    println!("{}", Matrix::new(vec![prediction1]));
    println!("{}", Matrix::new(vec![prediction2]));
    println!("{}", Matrix::new(vec![prediction3]));
    println!("{}", Matrix::new(vec![prediction4]));
}
