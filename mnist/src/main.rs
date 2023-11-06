mod load_mnist_data;

use linear_algebra::Matrix;
use load_mnist_data::load_data;
use neural_network;
use neural_network::NeuralNetwork;

use neural_network::activations::RELU;

fn main() {
    println!("Hello, world!");
    let activation = RELU;
    let loss_function = &|x: f64| x * x;

    // let mut network: NeuralNetwork =
    //     NeuralNetwork::new(vec![784, 10, 10], 0.3, activation, loss_function);
    let mut network = NeuralNetwork::new(vec![784, 10, 1], 0.3, activation, loss_function);
    network.load("./saved-network-mnist.json".to_string());
    let data: load_mnist_data::MNIST = load_data("./mnist/data");

    let inputs_train: Vec<Vec<f64>> = data.train_images;
    let target_train: Vec<Vec<f64>> = data.train_labels;
    let inputs_test: Vec<Vec<f64>> = data.test_images;
    let target_test: Vec<Vec<f64>> = data.test_labels;

    network.train(inputs_train, target_train, 1000);
    network.save("./saved-network-mnist.json".to_string());

    for i in 0..2 {
      let prediction1 = network.try_to_predict(inputs_test[i].clone());
      println!("{} prediction", Matrix::new(vec![prediction1]).transpose());
      println!("{} actual", Matrix::new(vec![target_test[i].clone()]).transpose());
    }
}
