use angstromgrad::Value;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn computes_forward_values_and_gradients() {
    let x = Value::from(2.0);
    let y = Value::from(3.0);
    let output = ((&x * &y) + &x).tanh();

    output.backward();

    let expected = 8.0_f64.tanh();
    assert_close(output.data(), expected);
    assert_close(x.grad(), 4.0 * (1.0 - expected.powi(2)));
    assert_close(y.grad(), 2.0 * (1.0 - expected.powi(2)));
}

#[test]
fn backpropagates_in_topological_order_through_shared_nodes() {
    let x = Value::from(2.0);
    let squared = &x * &x;
    let cubed = &squared * &x;
    let output = &squared + &cubed;

    output.backward();

    assert_close(output.data(), 12.0);
    assert_close(x.grad(), 16.0);
}

#[test]
fn treats_numerically_equal_nodes_as_distinct() {
    let first = Value::from(2.0);
    let second = Value::from(2.0);
    let first_branch = &first * &Value::from(3.0);
    let second_branch = &second * &Value::from(3.0);

    (first_branch + second_branch).backward();

    assert_close(first.grad(), 3.0);
    assert_close(second.grad(), 3.0);
}

#[test]
fn supports_reusing_an_operand() {
    let x = Value::from(4.0);
    let output = &x * &x + &x + &x;

    output.backward();

    assert_close(output.data(), 24.0);
    assert_close(x.grad(), 10.0);
}

#[test]
fn differentiates_both_parts_of_a_power() {
    let base = Value::from(2.0);
    let exponent = Value::from(3.0);
    let output = base.pow(&exponent);

    output.backward();

    assert_close(base.grad(), 12.0);
    assert_close(exponent.grad(), 8.0 * 2.0_f64.ln());
}

#[test]
fn repeated_backward_accumulates_only_fresh_leaf_gradients() {
    let x = Value::from(2.0);
    let squared = &x * &x;
    let output = &squared * &x;

    output.backward();
    output.backward();

    assert_close(x.grad(), 24.0);
}

#[test]
fn supports_owned_and_borrowed_arithmetic() {
    let x = Value::from(8.0);
    let y = Value::from(2.0);
    let output = (x.clone() / y.clone()) - (-&y) + (&x * y);

    assert_close(output.data(), 22.0);
}

#[test]
fn supports_arithmetic_with_scalars_on_either_side() {
    let x = Value::from(4.0);
    let output = 10.0 - &x + 12.0 / x.clone() + 2.0 * &x;

    output.backward();

    assert_close(output.data(), 17.0);
    assert_close(x.grad(), 0.25);
}

#[test]
fn differentiates_constant_powers_for_negative_bases() {
    let x = Value::from(-2.0);
    let output = x.powf(2.0);

    output.backward();

    assert_close(output.data(), 4.0);
    assert_close(x.grad(), -4.0);
}

#[test]
fn differentiates_activation_functions() {
    let relu_input = Value::from(2.0);
    let sigmoid_input = Value::from(0.0);
    let exp_input = Value::from(1.0);
    let output = relu_input.relu() + sigmoid_input.sigmoid() + exp_input.exp();

    output.backward();

    assert_close(output.data(), 2.5 + std::f64::consts::E);
    assert_close(relu_input.grad(), 1.0);
    assert_close(sigmoid_input.grad(), 0.25);
    assert_close(exp_input.grad(), std::f64::consts::E);
}

#[test]
fn relu_blocks_negative_gradients_and_sigmoid_is_stable() {
    let negative = Value::from(-2.0);
    let very_negative = Value::from(-1_000.0);
    let output = negative.relu() + very_negative.sigmoid();

    output.backward();

    assert_close(output.data(), 0.0);
    assert_close(negative.grad(), 0.0);
    assert_close(very_negative.grad(), 0.0);
}
