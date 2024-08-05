use std::cell::RefCell;
use std::rc::Rc;
use std::ops::{Add, Mul, Sub, Div};


#[derive(Debug, Clone, Copy)]
struct Value{
    data: f64,
    grad: f64,
    _backward: Rc<RefCell<dyn FnMut()>>,
    _prev: Vec<Rc<Value>>,   
    _op: Option<String>, 
}

impl Value {
    fn new(data: f64) -> Rc<Self> {
        Rc::new(Value {
            data,
            grad: 0.0,
            _backward: Rc::new(RefCell::new(|| {})),
            _prev: vec![],
            _op: None,
        })
    }
}

impl Add for Rc<Value>{
    type Output = Rc<Value>;
    fn add(self, other: Rc<Value>) -> Rc<Value> {
        let out = Value {
            data: self.data + other.data,
            grad: 0.0,
            _backward: Rc::new(RefCell::new(|| {})),
            _prev: vec![self.clone(), other.clone()]
        };

        let out_ref = Rc::new(out);

        {
            let out_clone = out_ref.clone();
            let backward = move || {
                self.grad += out_clone.grad;
                other.grad += out_clone.grad;
            };
            *out_ref._backward.borrow_mut() = Box::new(backward);
        }

        out_ref
    }
}

impl Mul for Rc<Value> {
    type Output = Rc<Value>;

    fn mul(self, other: Rc<Value>) -> Rc<Value> {
        let out = Value {
            data: self.data * other.data,
            grad: 0.0,
            _backward: Rc::new(RefCell::new(|| {})),
            _prev: vec![self.clone(), other.clone()],
        };

        let out_ref = Rc::new(out);
        {
            let out_clone = out_ref.clone();
            let backward = move || {
                self.grad += other.data * out_clone.grad;
                other.grad += self.data * out_clone.grad;
            };
            *out_ref._backward.borrow_mut() = Box::new(backward);
        }

        out_ref
    }
}

impl Value {
    fn backward(self: &Rc<Self>) {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(v: Rc<Value>, topo: &mut Vec<Rc<Value>>, visited: &mut std::collections::HashSet<*const Value> ) {
            if !visited.contains(&(v.as_ref() as * const _)) {
                visited.insert(v.as_ref() as *const _);
                for parent in &v._prev {
                    build_topo(parent.clone(), topo, visited);
                }
                topo.push(v);
            }
        }

        build_topo(self.clone(), &mut topo, &mut visited);

        self.grad = 1.0;

        for v in topo.into_iter().rev() {
            (v._backward.borrow_mut())();
        }
    }
}