use ndarray::Array2;
use neural_network::activations::SIGMOID;
use neural_network::NeuralNetwork;

fn main() {
    println!("Hello, world!");
    let activation = SIGMOID;

    let mut network = NeuralNetwork::new(vec![2, 3, 1], 0.3, activation);
    // network.load("./saved-network.json".to_string());

    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    network.train(inputs, target, 10000);
    network.save("./saved-network.json".to_string());

    let prediction1 =
        network.try_to_predict(Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap());
    let prediction2 =
        network.try_to_predict(Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap());
    let prediction3 =
        network.try_to_predict(Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap());
    let prediction4 =
        network.try_to_predict(Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap());

    println!("{}", prediction1);
    println!("{}", prediction2);
    println!("{}", prediction3);
    println!("{}", prediction4);
}
