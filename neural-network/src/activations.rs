use std::{f32::consts::E, collections::HashMap};

#[derive(Clone)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f32) -> f32,
    pub derivative: &'a dyn Fn(f32) -> f32,
}

pub const IDENTITY: Activation = Activation {
    function: &|x| x,
    derivative: &|_| 1.0,
};

pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|x| x * (1.0 - x),
};

pub const TANH: Activation = Activation {
    function: &|x| x.tanh(),
    derivative: &|x| 1.0 - (x.powi(2)),
};

pub const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};

pub const SOFTMAX: Activation = Activation {
    function: &|x| x.exp() / (1.0 + x.exp()),
    derivative: &|x| x * (1.0 - x),
};

pub fn get_activation_map() -> HashMap<String, &'static dyn Fn(f32) -> f32> {
  let mut activation_map: HashMap<String, &'static dyn Fn(f32) -> f32> = HashMap::new();
  activation_map.insert(String::from("IDENTITY"), IDENTITY.function);
  activation_map.insert(String::from("SIGMOID"), SIGMOID.function);
  activation_map.insert(String::from("TANH"), TANH.function);
  activation_map.insert(String::from("RELU"), RELU.function);
  activation_map.insert(String::from("SOFTMAX"), SOFTMAX.function);
  activation_map
}

pub fn get_activation_derivative_map() -> HashMap<String, &'static dyn Fn(f32) -> f32> {
  let mut activation_map: HashMap<String, &'static dyn Fn(f32) -> f32> = HashMap::new();
  activation_map.insert(String::from("IDENTITY"), IDENTITY.derivative);
  activation_map.insert(String::from("SIGMOID"), SIGMOID.derivative);
  activation_map.insert(String::from("TANH"), TANH.derivative);
  activation_map.insert(String::from("RELU"), RELU.derivative);
  activation_map.insert(String::from("SOFTMAX"), SOFTMAX.derivative);
  activation_map
}