use neural_network::layer::DenseLayer;
use neural_network::loss_functions::LossFunction;
use neural_network::{NeuralNetwork};

fn main() {
    let learning_rate: f32 = 0.2;
    let mut network: NeuralNetwork = NeuralNetwork::new(
        vec![
            Box::new(DenseLayer::new(2, 3, String::from("TANH"), learning_rate)),
            Box::new(DenseLayer::new(
                3,
                1,
                String::from("SIGMOID"),
                learning_rate,
            )),
        ],
        LossFunction::SquaredError,
    );

    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    network.train(inputs.clone(), target.clone(), 1000);
    // network.save("./saved-network.json".to_string());

    let mut testing_accuracy: f32 = 0.0;

    let inputs_test: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target_test: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    for i in 0..inputs_test.len() {
        let prediction = network.try_to_predict(inputs_test[i].clone());
        let actual = target_test[i];
        let predicted_rounded = if prediction[0] > 0.5 { 1.0 } else { 0.0 };
        if actual.eq(&predicted_rounded) {
            testing_accuracy += 1.0;
        }
        println!(
            "predicted {predicted} - should be {actual}",
            predicted = prediction[0],
            actual = actual
        )
    }
    println!(
        "Testing accuracy: {}",
        (testing_accuracy / target.len() as f32) * 100.0
    );
}
