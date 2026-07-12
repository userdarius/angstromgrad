# angstromgrad

`angstromgrad` is a small scalar reverse-mode automatic differentiation engine,
with a minimal multi-layer perceptron implementation. It is intended for learning
and experimentation rather than production numerical workloads.

```rust
use angstromgrad::{MLP, Value};

let model = MLP::with_seed(2, vec![4, 1], 42);
let inputs = vec![Value::from(2.0), Value::from(-1.0)];
let prediction = model.forward(&inputs).remove(0);
let target = Value::from(1.0);
let loss = (&prediction - &target).powf(2.0);

model.zero_grad();
loss.backward();
for parameter in model.parameters() {
    parameter.adjust(-0.05);
}
```

Run the test suite with:

```console
cargo test
```

Train a small network to learn XOR:

```console
cargo run --example xor
```

`Value` supports arithmetic with other values and directly with `f64` scalars,
along with `tanh`, `relu`, `sigmoid`, `exp`, differentiable `pow`, and
constant-exponent `powf` operations.
