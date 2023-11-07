mod load_mnist_data;

use load_mnist_data::load_data;
use neural_network;
use neural_network::NeuralNetwork;

use neural_network::activations::SIGMOID;

fn main() {
    println!("Hello, world!");
    let activation = SIGMOID;
    let loss_function = &|x: f64| x * x;

    // let mut network: NeuralNetwork =
    //     NeuralNetwork::new(vec![784, 10, 10], 0.03, activation, loss_function);
    let mut network = NeuralNetwork::new(vec![784, 10, 10], 0.03, activation, loss_function);
    network.load("./saved-network-mnist.json".to_string());
    let data: load_mnist_data::MNIST = load_data("./mnist/data");

    let inputs_train: Vec<Vec<f64>> = data.train_images;
    let target_train: Vec<Vec<f64>> = data.train_labels;
    let inputs_test: Vec<Vec<f64>> = data.test_images;
    let target_test: Vec<Vec<f64>> = data.test_labels;

    network.train(inputs_train, target_train, 1000);
    network.save("./saved-network-mnist.json".to_string());

    let mut testing_precision: f64 = 0.0;

    for i in 0..inputs_test.len() {
        let prediction1 = network.try_to_predict(inputs_test[i].clone());
        let actual = NeuralNetwork::get_max_value_index(target_test[i].clone());
        let predicted = NeuralNetwork::get_max_value_index(prediction1);
        if actual.eq(&predicted) {
            testing_precision += 1.0;
        }
        println!("{} Prediction {} x {} Actual", actual.eq(&predicted), predicted, actual);
    }
    println!("Testing precision: {}", (testing_precision / inputs_test.len() as f64) * 100.0)
}
