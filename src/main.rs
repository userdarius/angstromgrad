use std::rc::Rc;
use std::cell::RefCell;
use value::Value; // Import the Value type from the value module

mod value;

fn main() {
    let x = Value::new(2.0);
    let y = Value::new(3.0);

    let z = x.clone() + y.clone() * x.clone();

    z.backward();

    println!("z: {:?}", z);
    println!("x.grad: {}", x.grad);
    println!("y.grad: {}", y.grad);
}