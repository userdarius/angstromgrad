use crate::engine::Value;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Shared behavior for trainable components.
pub trait Module {
    /// Returns all trainable parameters owned by this module.
    fn parameters(&self) -> Vec<Value>;

    /// Clears the gradient on every trainable parameter.
    fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.zero_grad();
        }
    }
}

#[derive(Clone, Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    /// Constructs a neuron with weights and bias sampled uniformly from `-1..1`.
    pub fn new(input_count: usize) -> Self {
        Self::with_rng(input_count, &mut rand::rng())
    }

    /// Constructs a reproducibly initialized neuron.
    pub fn with_seed(input_count: usize, seed: u64) -> Self {
        Self::with_rng(input_count, &mut StdRng::seed_from_u64(seed))
    }

    fn with_rng<R: Rng + ?Sized>(input_count: usize, rng: &mut R) -> Self {
        let mut random_value = || Value::from(rng.random_range(-1.0..1.0));
        let weights = (0..input_count).map(|_| random_value()).collect();
        let bias = random_value().add_label("bias");
        Self { weights, bias }
    }

    /// Computes `tanh(weights · inputs + bias)`.
    ///
    /// # Panics
    ///
    /// Panics when the number of inputs does not match the neuron's weights.
    pub fn forward(&self, inputs: &[Value]) -> Value {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "expected {} inputs, received {}",
            self.weights.len(),
            inputs.len()
        );

        let activation = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(weight, input)| weight * input)
            .fold(self.bias.clone(), |sum, product| sum + product);
        activation.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        <Self as Module>::parameters(self)
    }

    pub fn zero_grad(&self) {
        <Self as Module>::zero_grad(self);
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        self.weights
            .iter()
            .cloned()
            .chain(std::iter::once(self.bias.clone()))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Constructs a layer containing `output_count` neurons.
    pub fn new(input_count: usize, output_count: usize) -> Self {
        Self::with_rng(input_count, output_count, &mut rand::rng())
    }

    /// Constructs a reproducibly initialized layer.
    pub fn with_seed(input_count: usize, output_count: usize, seed: u64) -> Self {
        Self::with_rng(input_count, output_count, &mut StdRng::seed_from_u64(seed))
    }

    fn with_rng<R: Rng + ?Sized>(input_count: usize, output_count: usize, rng: &mut R) -> Self {
        let neurons = (0..output_count)
            .map(|_| Neuron::with_rng(input_count, rng))
            .collect();
        Self { neurons }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        <Self as Module>::parameters(self)
    }

    pub fn zero_grad(&self) {
        <Self as Module>::zero_grad(self);
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(Module::parameters).collect()
    }
}

#[derive(Clone, Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Constructs a multi-layer perceptron.
    ///
    /// `layer_sizes` contains the output width of each successive layer.
    pub fn new(input_count: usize, layer_sizes: Vec<usize>) -> Self {
        Self::with_rng(input_count, &layer_sizes, &mut rand::rng())
    }

    /// Constructs a reproducibly initialized multi-layer perceptron.
    pub fn with_seed(input_count: usize, layer_sizes: Vec<usize>, seed: u64) -> Self {
        Self::with_rng(input_count, &layer_sizes, &mut StdRng::seed_from_u64(seed))
    }

    fn with_rng<R: Rng + ?Sized>(input_count: usize, layer_sizes: &[usize], rng: &mut R) -> Self {
        let mut previous_size = input_count;
        let layers = layer_sizes
            .iter()
            .map(|&size| {
                let layer = Layer::with_rng(previous_size, size, rng);
                previous_size = size;
                layer
            })
            .collect();
        Self { layers }
    }

    /// Performs a forward pass. Both owned vectors and slices are accepted.
    pub fn forward(&self, inputs: impl AsRef<[Value]>) -> Vec<Value> {
        let mut outputs = inputs.as_ref().to_vec();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Value> {
        <Self as Module>::parameters(self)
    }

    pub fn zero_grad(&self) {
        <Self as Module>::zero_grad(self);
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(Module::parameters).collect()
    }
}
