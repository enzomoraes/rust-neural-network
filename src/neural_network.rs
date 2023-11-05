use crate::linear_algebra::Matrix;

pub struct Layer {
    inputs: usize,
    outputs: usize,
    weights: Matrix,
    biases: Matrix,
    inputs_matrix: Matrix,
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
            inputs_matrix: Matrix::zero(inputs, 1),
        };
    }

    pub fn add_to_weights(&mut self, matrix: &Matrix) {
        self.weights = self.weights.add(&matrix);
    }

    pub fn add_to_biases(&mut self, matrix: &Matrix) {
        self.biases = self.biases.add(&matrix);
    }

    pub fn weigh_inputs(&self, inputs: &Matrix) -> Matrix {
        return self.weights.multiply(&inputs).add(&self.biases);
    }

    fn set_inputs_matrix(&mut self, clone: Matrix) {
        self.inputs_matrix = clone;
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

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, target: &Vec<Vec<f64>>, epochs: usize) {
        for i in 1..=epochs {
            if i % (epochs) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let predictions = self.feed_forward(&inputs[j]);
                self.back_propagate(&predictions, &target[j]);
            }
        }
    }

    pub fn try_to_predict(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        return self.feed_forward(inputs);
    }

    fn feed_forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0].inputs {
            panic!("Inputs length does not match with neural network");
        }

        let mut data: Matrix = Matrix::new(vec![inputs.to_vec()]).transpose();

        for i in 0..self.layers.len() {
            self.layers[i].set_inputs_matrix(data.clone());
            data = self.layers[i]
                .weigh_inputs(&data)
                .apply_function(&self.activation.function);
        }

        return data.transpose().data[0].to_owned();
    }

    fn back_propagate(&mut self, predictions: &Vec<f64>, target: &Vec<f64>) {
        let loss = self.loss(
            &Matrix::new(vec![target.to_owned()]),
            &Matrix::new(vec![predictions.to_owned()]),
        );
        let size: usize = self.layers.len();
        let mut gradient = self.gradient(&loss.transpose());

        for i in (1..size).rev() {
            gradient = gradient
                .apply_function(&self.activation.derivative)
                .apply_function(&|x| x * self.learning_rate);

            let gradient_weights = &gradient.multiply(&self.layers[i].inputs_matrix.transpose());
            self.layers[i].add_to_weights(gradient_weights);
            self.layers[i].add_to_biases(&gradient.apply_function(&|x| x * self.learning_rate));
        }
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
