use crate::linear_algebra::Matrix;

pub struct Layer {
    inputs: usize,
    outputs: usize,
    weights: Matrix,
    biases: Matrix,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let weights: Matrix = Matrix::random(outputs, inputs);
        let biases: Matrix = Matrix::random(outputs, 1);

        return Layer {
            inputs,
            outputs,
            weights,
            biases,
        };
    }

    pub fn weigh_inputs(&self, inputs: &Matrix) -> Matrix {
        return self.weights.multiply(&inputs).add(&self.biases);
    }
}

pub struct Activation {
    pub function: Box<dyn Fn(f64) -> f64>,
    pub derivative: Box<dyn Fn(f64) -> f64>,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    activation: Activation,
    loss_function: Box<dyn Fn(f64) -> f64>,
}

impl NeuralNetwork {
    pub fn new(
        neurons_per_layer: Vec<usize>,
        learning_rate: f64,
        activation: Activation,
        loss_function: Box<dyn Fn(f64) -> f64>,
    ) -> NeuralNetwork {
        let mut layers: Vec<Layer> = vec![];

        for i in 1..neurons_per_layer.len() {
            layers.push(Layer::new(neurons_per_layer[i - 1], neurons_per_layer[i]));
        }

        return NeuralNetwork {
            layers,
            learning_rate,
            activation,
            loss_function,
        };
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Matrix {
        if inputs.len() != self.layers[0].inputs {
            panic!("Inputs length does not match with neural network");
        }

        let mut data: Matrix = Matrix::new(vec![inputs.to_vec()]).transpose();

        for i in 0..self.layers.len() {
            data = self.layers[i]
                .weigh_inputs(&data)
                .apply_function(&self.activation.function);
        }

        return data;
    }

    fn loss(&self, target: &Matrix, predictions: &Matrix) -> Matrix {
        return target
            .subtract(&predictions)
            .apply_function(&self.loss_function);
    }

    fn gradient(&self, matrix: &Matrix) -> Matrix {
        return matrix.apply_function(&self.activation.derivative);
    }
}
