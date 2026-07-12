use angstromgrad::{Value, MLP};

fn main() {
    let model = MLP::with_seed(2, vec![4, 1], 42);
    let samples = [
        ([-1.0, -1.0], -1.0),
        ([-1.0, 1.0], 1.0),
        ([1.0, -1.0], 1.0),
        ([1.0, 1.0], -1.0),
    ];

    for step in 0..500 {
        let predictions: Vec<_> = samples
            .iter()
            .map(|(inputs, _)| model.forward(inputs.map(Value::from)).remove(0))
            .collect();

        let loss: Value = predictions
            .iter()
            .zip(&samples)
            .map(|(prediction, (_, target))| (prediction - *target).powf(2.0))
            .sum::<Value>()
            / samples.len() as f64;

        model.zero_grad();
        loss.backward();
        for parameter in model.parameters() {
            parameter.adjust(-0.1);
        }

        if step % 100 == 0 || step == 499 {
            println!("step {step:>3}: loss = {:.6}", loss.data());
        }
    }

    println!("\npredictions:");
    for (inputs, target) in samples {
        let prediction = model.forward(inputs.map(Value::from)).remove(0);
        println!(
            "{inputs:?} -> {:+.3} (target {target:+.0})",
            prediction.data()
        );
    }
}
