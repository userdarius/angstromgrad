#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod engine;
pub use crate::engine::Value;

pub mod nn;
pub use crate::nn::{Layer, Module, Neuron, MLP};
