use neural_network::NeuralNetwork;

use linear_algebra::Matrix;

fn main() {
    println!("Hello, world!");

    let mut network = NeuralNetwork::new(vec![2, 3, 1]);
    network.load("./saved-network.json".to_string());

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
