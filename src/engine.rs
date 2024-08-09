use std::cell::{Ref, RefCell};
use std::iter::Sum;
use std::ops::{Add, Deref, Mul, Neg, Sub};
use std::rc::Rc;

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

type PropagateFn = fn(value: &Ref<_Value>);

pub struct _Value {
    data: f64,
    grad: f64,
    _op: Option<String>,
    _prev: Vec<Value>,
    propagate: Option<PropagateFn>,
    label: Option<String>,
}

impl _Value {
    fn new(
        data: f64,
        label: Option<String>,
        op: Option<String>,
        prev: Vec<Value>,
        propagate: Option<PropagateFn>,
    ) -> _Value {
        _Value {
            data, // the actual numerical value
            grad: 0.0, // gradient of the value with respect to some loss
            label, // optional label for the value
            _op: op, // optional string to describe the operation that created this value
            _prev: prev, // vector of previous _Value instances linked to this value
            propagate, // optional function for propagating gradients back through the network
        }
    }
}

// check if two values are equal by comparing their attributes
impl PartialEq for _Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.grad == other.grad
            && self.label == other.label
            && self._op == other._op
            && self._prev == other._prev
    }
}

impl Eq for _Value {}

impl Hash for _Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.grad.to_bits().hash(state);
        self.label.hash(state);
        self._op.hash(state);
        self._prev.hash(state);
    }
}

// Don't really know why we need this 
impl Debug for _Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("_Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("label", &self.label)
            .field("_op", &self._op)
            .field("_prev", &self._prev)
            .finish()
    }
}

// Wrapper around _Value to allow for multiple references to the same _Value instance 
// while allowing for interior mutability by using Rc<RefCell<...>>  
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<_Value>>);

impl Value {
   
    pub fn from<T>(t: T) -> Value
    where
        T: Into<Value>,
    {
        t.into()
    }

    // A new `Value` that wraps the given `_Value` in `Rc<RefCell<_Value>>`.
    fn new(value: _Value) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    // recursively applies the propagation function defined in `_Value` nodes.
    pub fn backward(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.borrow_mut().grad = 1.0;

        fn _backward(visited: &mut HashSet<Value>, value: &Value) {
            if !visited.contains(&value) {
                visited.insert(value.clone());

                let borrowed_value = value.borrow();
                if let Some(propagate_fn) = borrowed_value.propagate {
                    propagate_fn(&borrowed_value);
                }

                for child_id in &value.borrow()._prev {
                    _backward(visited, child_id);
                }
            }
        }

        _backward(&mut visited, self);
    }
    
    pub fn pow(&self, other: &Value) -> Value {
        let result = self.borrow().data.powf(other.borrow().data);

        let propagate_fn: PropagateFn = |value| {
            let mut base = value._prev[0].borrow_mut();
            let power = value._prev[1].borrow();
            base.grad += power.data * (base.data.powf(power.data - 1.0)) * value.grad;
        };

        Value::new(_Value::new(
            result,
            None,
            Some("^".to_string()),
            vec![self.clone(), other.clone()],
            Some(propagate_fn),
        ))
    }

    pub fn tanh(&self) -> Value {
        let result = self.borrow().data.tanh();

        let propagate_fn: PropagateFn = |value| {
            let mut _prev = value._prev[0].borrow_mut();
            _prev.grad += (1.0 - value.data.powf(2.0)) * value.grad;
        };

        Value::new(_Value::new(
            result,
            None,
            Some("tanh".to_string()),
            vec![self.clone()],
            Some(propagate_fn),
        ))
    }

    pub fn add_label(self, label: &str) -> Value {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn zero_grad(&self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn adjust(&self, factor: f64) {
        let mut value = self.borrow_mut();
        value.data += factor * value.grad;
    }
}

// Create a hashed value for the `Value` instance based on the inner `_Value` instance.
impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

// Provides dereference access to the Rc<RefCell<_Value>>. A reference to the inner storage of the `Value`.
impl Deref for Value {
    type Target = Rc<RefCell<_Value>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
// Converts a type that can be cast into a floating point number directly into a `Value`.
impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(_Value::new(t.into(), None, None, Vec::new(), None))
    }
}

// Allows to add two Value instances without consuming them. 
// Instead, it takes references to self and other, allowing to reuse the original Value instances after the addition.
impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;
    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data + b.borrow().data;

    let propagate_fn: PropagateFn = |value| {
        let mut first = value._prev[0].borrow_mut();
        let mut second = value._prev[1].borrow_mut();

        first.grad += value.grad;
        second.grad += value.grad;
    };

    Value::new(_Value::new(
        result,
        None,
        Some("+".to_string()),
        vec![a.clone(), b.clone()],
        Some(propagate_fn),
    ))
}


impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        mul(self, other)
    }
}

// the gradient of the result is multiplied by the other operand's value before being propagated back.
fn mul(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data * b.borrow().data;

    let propagate_fn: PropagateFn = |value| {
        let mut first = value._prev[0].borrow_mut();
        let mut second = value._prev[1].borrow_mut();

        first.grad += second.data * value.grad;
        second.grad += first.data * value.grad;
    };

    Value::new(_Value::new(
        result,
        None,
        Some("*".to_string()),
        vec![a.clone(), b.clone()],
        Some(propagate_fn),
    ))
}

impl<'a> Neg for &'a Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        mul(self, &Value::from(-1))
    }
}

// Sums all elements in an iterator over `Value` and returns a single `Value` representing the sum.
impl Sum for Value {
   
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut sum = Value::from(0.0);
        loop {
            let val = iter.next();
            if val.is_none() {
                break;
            }

            sum = sum + val.unwrap();
        }
        sum
    }
}
