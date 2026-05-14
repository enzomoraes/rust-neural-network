mod load_mnist_data;

use std::env;

use load_mnist_data::load_data;
use neural_network::layer::DenseLayer;
use neural_network::loss_functions::LossFunction;
use neural_network::savable_neural_network::SavableNeuralNetwork;
use neural_network::NeuralNetwork;

fn main() {
    let block_size = 128;
    env::set_var("BLOCK_SIZE", block_size.to_string());

    let learning_rate: f32 = 0.01;
    let layers_array = [
        DenseLayer::new(784, 128, String::from("RELU"), learning_rate),
        DenseLayer::new(128, 32, String::from("RELU"), learning_rate),
        DenseLayer::new(32, 10, String::from("SIGMOID"), learning_rate),
    ];
    println!(
        "Started - learning rate {learning_rate} - using block_size for tiling of {block_size}",
        learning_rate = learning_rate,
        block_size = block_size
    );
    println!("Network Architecture:");
    for layer in layers_array.iter() {
        println!("  {}", layer);
    }

    let layers: Vec<Box<dyn neural_network::layer::Layer>> = layers_array
        .into_iter()
        .map(|l| Box::new(l) as Box<dyn neural_network::layer::Layer>)
        .collect();

    let mut network: NeuralNetwork = NeuralNetwork::new(layers, LossFunction::SquaredError);

    let data: load_mnist_data::MNIST = load_data("./mnist/data");

    let inputs_train: Vec<Vec<f32>> = data.train_images;
    let target_train: Vec<Vec<f32>> = data.train_labels;
    let inputs_test: Vec<Vec<f32>> = data.test_images;
    let target_test: Vec<Vec<f32>> = data.test_labels;

    network.train(inputs_train, target_train, 10);
    network.save("./saved-network-mnist.json".to_string());

    let mut testing_accuracy: f32 = 0.0;

    for i in 0..inputs_test.len() {
        let prediction1 = network.try_to_predict(inputs_test[i].clone());
        let actual = NeuralNetwork::get_max_value_index(target_test[i].clone());
        let predicted = NeuralNetwork::get_max_value_index(prediction1);
        if actual.eq(&predicted) {
            testing_accuracy += 1.0;
        }
    }
    println!(
        "Testing accuracy: {}",
        (testing_accuracy / inputs_test.len() as f32) * 100.0
    );
}
