use crate::linear_algebra::Matrix;

pub struct NeuralNetwork {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    learning_rate: f64,
    activation: Box<dyn Fn(f64) -> f64>,
}

impl NeuralNetwork {
    pub fn new(
        layers: Vec<usize>,
        learning_rate: f64,
        activation: Box<dyn Fn(f64) -> f64>,
    ) -> NeuralNetwork {
        let mut weights: Vec<Matrix> = vec![];
        let mut biases: Vec<Matrix> = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        return NeuralNetwork {
            layers,
            weights,
            biases,
            learning_rate,
            activation,
        };
    }

    pub fn feed_forward(&self, inputs: Vec<f64>) -> Matrix {
        if inputs.len() != self.layers[0] {
            panic!("Inputs length does not match with neural network");
        }

        let mut data: Matrix = Matrix::new(vec![inputs]).transpose();

        for i in 0..self.layers.len() - 1 {
            data = self.weights[i]
                .multiply(&data)
                .add(&self.biases[i])
                .apply_function(&self.activation);
        }

        return data;
    }
}
