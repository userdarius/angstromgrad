use angstromgrad::{Neuron, Value, MLP};

#[test]
fn seeded_initialization_is_reproducible() {
    let first = MLP::with_seed(2, vec![3, 1], 7);
    let second = MLP::with_seed(2, vec![3, 1], 7);

    let first_values: Vec<_> = first.parameters().iter().map(Value::data).collect();
    let second_values: Vec<_> = second.parameters().iter().map(Value::data).collect();
    assert_eq!(first_values, second_values);
}

#[test]
fn mlp_has_the_expected_shape_and_parameter_count() {
    let model = MLP::with_seed(2, vec![3, 1], 11);
    let output = model.forward([Value::from(1.0), Value::from(-2.0)]);

    assert_eq!(output.len(), 1);
    assert_eq!(model.parameters().len(), 13);
}

#[test]
fn zero_grad_clears_all_parameter_gradients() {
    let model = MLP::with_seed(2, vec![2, 1], 13);
    let output = model.forward([Value::from(0.5), Value::from(-1.0)]);
    output[0].backward();
    assert!(model
        .parameters()
        .iter()
        .any(|parameter| parameter.grad() != 0.0));

    model.zero_grad();

    assert!(model
        .parameters()
        .iter()
        .all(|parameter| parameter.grad() == 0.0));
}

#[test]
fn zero_input_neuron_uses_its_bias() {
    let neuron = Neuron::with_seed(0, 17);
    let output = neuron.forward(&[]);
    assert!(output.data().is_finite());
}

#[test]
#[should_panic(expected = "expected 2 inputs, received 1")]
fn neuron_rejects_the_wrong_input_count() {
    Neuron::with_seed(2, 19).forward(&[Value::from(1.0)]);
}
