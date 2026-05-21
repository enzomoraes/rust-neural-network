mod data_augmentation;
mod image_export;
mod load_mnist_data;

use data_augmentation::{augment_mnist, AugmentationConfig};
use image_export::save_augmented_samples;
use load_mnist_data::load_data;
use neural_network::activations::ActivationFunction;
use neural_network::layer::DenseLayer;
use neural_network::loss_functions::LossFunction;
use neural_network::savable_neural_network::SavableNeuralNetwork;
use neural_network::NeuralNetwork;

fn main() {
    let learning_rate: f32 = 0.01;
    let layers_array = [
        DenseLayer::new(784, 128, ActivationFunction::Relu, learning_rate),
        DenseLayer::new(128, 32, ActivationFunction::Relu, learning_rate),
        DenseLayer::new(32, 10, ActivationFunction::Sigmoid, learning_rate),
    ];
    println!(
        "Started - learning rate {learning_rate}",
        learning_rate = learning_rate
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
    // let mut network: NeuralNetwork = NeuralNetwork::load("./saved-network-mnist.json".to_string());

    let data: load_mnist_data::MNIST = load_data("./mnist/data");

    let inputs_test: Vec<Vec<f32>> = data.test_images;
    let target_test: Vec<Vec<f32>> = data.test_labels;

    println!("\nAugmenting training data with Gaussian noise and rotations...");
    let (inputs_train, target_train) = augment_mnist(
        &data.train_images,
        &data.train_labels,
        &AugmentationConfig {
            noise_stddev: 0.05,
            rotation_angles: vec![10.0, -10.0],
        },
    );
    println!("Original training images: {}", data.train_images.len());
    println!("Augmented training images: {}", inputs_train.len());

    println!("\nExporting sample augmented images for visualization...");
    match save_augmented_samples(&inputs_train, 5, "./augmented_samples") {
        Ok(()) => println!("✓ Images saved successfully"),
        Err(e) => eprintln!("✗ Error saving images: {}", e),
    }

    network.train_with_batches(inputs_train, target_train, 10, 32);
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
