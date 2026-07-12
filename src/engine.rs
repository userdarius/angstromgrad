use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

type PropagateFn = fn(&Node);

struct Node {
    data: f64,
    grad: f64,
    op: Option<&'static str>,
    prev: Vec<Value>,
    propagate: Option<PropagateFn>,
    label: Option<String>,
}

impl Node {
    fn new(
        data: f64,
        label: Option<String>,
        op: Option<&'static str>,
        prev: Vec<Value>,
        propagate: Option<PropagateFn>,
    ) -> Self {
        Self {
            data,
            grad: 0.0,
            label,
            op,
            prev,
            propagate,
        }
    }
}

/// A scalar value and its node in a dynamically constructed computation graph.
///
/// Cloning a `Value` is cheap: clones refer to the same graph node and therefore
/// observe the same data and gradient.
#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

impl Value {
    fn new(node: Node) -> Self {
        Self(Rc::new(RefCell::new(node)))
    }

    /// Runs reverse-mode automatic differentiation from this value.
    ///
    /// Gradients accumulate into leaf nodes. Call [`Value::zero_grad`] (or
    /// [`crate::nn::Module::zero_grad`] for a network) before a new optimization
    /// step when accumulation is not desired. Intermediate gradients are reset
    /// on each call so reusing a graph does not propagate stale values.
    pub fn backward(&self) {
        fn build_topology(value: &Value, visited: &mut HashSet<usize>, topology: &mut Vec<Value>) {
            let identity = Rc::as_ptr(&value.0) as usize;
            if !visited.insert(identity) {
                return;
            }

            let predecessors = value.0.borrow().prev.clone();
            for predecessor in predecessors {
                build_topology(&predecessor, visited, topology);
            }
            topology.push(value.clone());
        }

        let mut topology = Vec::new();
        build_topology(self, &mut HashSet::new(), &mut topology);

        for value in &topology {
            let mut node = value.0.borrow_mut();
            if node.propagate.is_some() {
                node.grad = 0.0;
            }
        }

        let mut root = self.0.borrow_mut();
        if root.propagate.is_some() {
            root.grad = 1.0;
        } else {
            root.grad += 1.0;
        }
        drop(root);

        for value in topology.into_iter().rev() {
            let node = value.0.borrow();
            if let Some(propagate) = node.propagate {
                propagate(&node);
            }
        }
    }

    /// Raises this value to a differentiable scalar exponent.
    pub fn pow(&self, exponent: &Value) -> Value {
        let result = self.data().powf(exponent.data());

        let propagate: PropagateFn = |node| {
            let grad = node.grad;
            let base_data = node.prev[0].data();
            let exponent_data = node.prev[1].data();

            node.prev[0].0.borrow_mut().grad +=
                exponent_data * base_data.powf(exponent_data - 1.0) * grad;
            node.prev[1].0.borrow_mut().grad += node.data * base_data.ln() * grad;
        };

        Value::new(Node::new(
            result,
            None,
            Some("pow"),
            vec![self.clone(), exponent.clone()],
            Some(propagate),
        ))
    }

    /// Raises this value to a constant floating-point exponent.
    pub fn powf(&self, exponent: f64) -> Value {
        let result = self.data().powf(exponent);
        let propagate: PropagateFn = |node| {
            let exponent = node.prev[1].data();
            let base = node.prev[0].data();
            node.prev[0].0.borrow_mut().grad += exponent * base.powf(exponent - 1.0) * node.grad;
        };

        Value::new(Node::new(
            result,
            None,
            Some("powf"),
            vec![self.clone(), Value::from(exponent)],
            Some(propagate),
        ))
    }

    /// Applies the hyperbolic tangent activation function.
    pub fn tanh(&self) -> Value {
        let result = self.data().tanh();

        let propagate: PropagateFn = |node| {
            node.prev[0].0.borrow_mut().grad += (1.0 - node.data.powi(2)) * node.grad;
        };

        Value::new(Node::new(
            result,
            None,
            Some("tanh"),
            vec![self.clone()],
            Some(propagate),
        ))
    }

    /// Applies the rectified linear unit activation function.
    pub fn relu(&self) -> Value {
        let result = self.data().max(0.0);
        let propagate: PropagateFn = |node| {
            if node.data > 0.0 {
                node.prev[0].0.borrow_mut().grad += node.grad;
            }
        };

        Value::new(Node::new(
            result,
            None,
            Some("relu"),
            vec![self.clone()],
            Some(propagate),
        ))
    }

    /// Applies the logistic sigmoid activation function.
    pub fn sigmoid(&self) -> Value {
        let input = self.data();
        let result = if input >= 0.0 {
            1.0 / (1.0 + (-input).exp())
        } else {
            let exp = input.exp();
            exp / (1.0 + exp)
        };
        let propagate: PropagateFn = |node| {
            node.prev[0].0.borrow_mut().grad += node.data * (1.0 - node.data) * node.grad;
        };

        Value::new(Node::new(
            result,
            None,
            Some("sigmoid"),
            vec![self.clone()],
            Some(propagate),
        ))
    }

    /// Applies the exponential function.
    pub fn exp(&self) -> Value {
        let result = self.data().exp();
        let propagate: PropagateFn = |node| {
            node.prev[0].0.borrow_mut().grad += node.data * node.grad;
        };

        Value::new(Node::new(
            result,
            None,
            Some("exp"),
            vec![self.clone()],
            Some(propagate),
        ))
    }

    /// Adds a human-readable label, useful while inspecting a graph.
    pub fn add_label(self, label: impl Into<String>) -> Value {
        self.0.borrow_mut().label = Some(label.into());
        self
    }

    /// Returns the scalar's current numerical value.
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    /// Returns the accumulated gradient.
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    /// Clears this node's accumulated gradient.
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    /// Applies an in-place gradient update: `data += factor * grad`.
    pub fn adjust(&self, factor: f64) {
        let mut node = self.0.borrow_mut();
        node.data += factor * node.grad;
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.0.borrow();
        formatter
            .debug_struct("Value")
            .field("data", &node.data)
            .field("grad", &node.grad)
            .field("label", &node.label)
            .field("op", &node.op)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data().fmt(formatter)
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(value: T) -> Self {
        Value::new(Node::new(value.into(), None, None, Vec::new(), None))
    }
}

fn add(left: &Value, right: &Value) -> Value {
    let result = left.data() + right.data();
    let propagate: PropagateFn = |node| {
        let grad = node.grad;
        node.prev[0].0.borrow_mut().grad += grad;
        node.prev[1].0.borrow_mut().grad += grad;
    };

    Value::new(Node::new(
        result,
        None,
        Some("+"),
        vec![left.clone(), right.clone()],
        Some(propagate),
    ))
}

fn mul(left: &Value, right: &Value) -> Value {
    let left_data = left.data();
    let right_data = right.data();
    let propagate: PropagateFn = |node| {
        let grad = node.grad;
        let left_data = node.prev[0].data();
        let right_data = node.prev[1].data();
        node.prev[0].0.borrow_mut().grad += right_data * grad;
        node.prev[1].0.borrow_mut().grad += left_data * grad;
    };

    Value::new(Node::new(
        left_data * right_data,
        None,
        Some("*"),
        vec![left.clone(), right.clone()],
        Some(propagate),
    ))
}

fn sub(left: &Value, right: &Value) -> Value {
    add(left, &-right)
}

fn div(left: &Value, right: &Value) -> Value {
    mul(left, &right.powf(-1.0))
}

macro_rules! impl_binary_operator {
    ($trait:ident, $method:ident, $function:ident) => {
        impl $trait<Value> for Value {
            type Output = Value;

            fn $method(self, rhs: Value) -> Self::Output {
                $function(&self, &rhs)
            }
        }

        impl $trait<&Value> for Value {
            type Output = Value;

            fn $method(self, rhs: &Value) -> Self::Output {
                $function(&self, rhs)
            }
        }

        impl $trait<Value> for &Value {
            type Output = Value;

            fn $method(self, rhs: Value) -> Self::Output {
                $function(self, &rhs)
            }
        }

        impl $trait<&Value> for &Value {
            type Output = Value;

            fn $method(self, rhs: &Value) -> Self::Output {
                $function(self, rhs)
            }
        }
    };
}

impl_binary_operator!(Add, add, add);
impl_binary_operator!(Mul, mul, mul);
impl_binary_operator!(Sub, sub, sub);
impl_binary_operator!(Div, div, div);

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(&self, &Value::from(-1.0))
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(self, &Value::from(-1.0))
    }
}

macro_rules! impl_scalar_operator {
    ($trait:ident, $method:ident, $function:ident) => {
        impl $trait<f64> for Value {
            type Output = Value;

            fn $method(self, rhs: f64) -> Self::Output {
                $function(&self, &Value::from(rhs))
            }
        }

        impl $trait<f64> for &Value {
            type Output = Value;

            fn $method(self, rhs: f64) -> Self::Output {
                $function(self, &Value::from(rhs))
            }
        }

        impl $trait<Value> for f64 {
            type Output = Value;

            fn $method(self, rhs: Value) -> Self::Output {
                $function(&Value::from(self), &rhs)
            }
        }

        impl $trait<&Value> for f64 {
            type Output = Value;

            fn $method(self, rhs: &Value) -> Self::Output {
                $function(&Value::from(self), rhs)
            }
        }
    };
}

impl_scalar_operator!(Add, add, add);
impl_scalar_operator!(Mul, mul, mul);
impl_scalar_operator!(Sub, sub, sub);
impl_scalar_operator!(Div, div, div);

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Value::from(0.0), |sum, value| sum + value)
    }
}

impl<'a> Sum<&'a Value> for Value {
    fn sum<I: Iterator<Item = &'a Value>>(iter: I) -> Self {
        iter.fold(Value::from(0.0), |sum, value| sum + value)
    }
}
