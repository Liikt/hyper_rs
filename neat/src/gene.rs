use rand::{random, thread_rng, Rng};
use rand_distr::{Normal, Distribution};

use crate::config;
use crate::aggregation::AggregationFunction;
use crate::activation::ActivationFunction;

#[derive(Debug, Clone, Copy)]
pub(crate) struct NodeGene {
    id: usize,

    bias: f64,
    response: f64,

    activation: ActivationFunction,
    aggregation: AggregationFunction
}

impl NodeGene {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            bias: 1.0,
            response: 1.0,
            aggregation: AggregationFunction::Sum,
            activation: ActivationFunction::Sigmoid,
        }
    }

    pub fn activate(&self, vals: &[f64]) -> f64 {
        self.bias + self.response * 
            self.activation.activate(self.aggregate(vals))
    }

    pub fn aggregate(&self, vals: &[f64]) -> f64 {
        self.aggregation.aggregate(vals)
    }

    pub fn mutate(&mut self) {
        let mut rng = thread_rng();

        let r = rng.gen_range(0.0..=1.0);
        if r < config::BIAS_MUTATE_RATE {
            let normal = Normal::new(0.0, config::BIAS_MUTATE_POWER).unwrap();
            self.bias = (self.bias + normal.sample(&mut rng))
                .clamp(config::BIAS_MIN_VALUE, config::BIAS_MAX_VALUE);
        } else if r < config::BIAS_MUTATE_RATE + config::BIAS_REPLACE_RATE {
            self.bias = (random::<f64>() % 
                (config::BIAS_MIN_VALUE + config::BIAS_MAX_VALUE)) - 
                config::BIAS_MIN_VALUE;
        }

        let r = rng.gen_range(0.0..=1.0);
        if r < config::RESPONSE_MUTATE_RATE {
            let normal = Normal::new(0.0, config::RESPONSE_MUTATE_POWER)
                .unwrap();
            self.response = (self.response + normal.sample(&mut rng))
                .clamp(config::RESPONSE_MIN_VALUE, config::RESPONSE_MAX_VALUE);
        } else if r < config::RESPONSE_MUTATE_RATE + 
                config::RESPONSE_REPLACE_RATE {
            self.response = (random::<f64>() % 
                (config::RESPONSE_MIN_VALUE + config::RESPONSE_MAX_VALUE)) - 
                config::RESPONSE_MIN_VALUE;
        }

        let r = rng.gen_range(0.0..=1.0);
        if r < config::ACTIVATION_MUT_PROB {
            self.activation = random();
        }

        let r = rng.gen_range(0.0..=1.0);
        if r < config::AGGREGATION_MUT_PROB {
            self.aggregation = random();
        }
    }

    pub fn crossover(&self, other: &Self) -> Self {
        let b = random::<f64>() % 1.0;
        let ag = random::<f64>() % 1.0;
        let ac = random::<f64>() % 1.0;
        let mut ret = Self::new(self.id);

        ret.bias = if b < 0.5 { other.bias } else { self.bias };
        ret.activation = 
            if ac < 0.5 { other.activation } else { self.activation };
        ret.aggregation = 
            if ag < 0.5 { other.aggregation } else { self.aggregation };
        ret
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let mut d = (self.bias - other.bias).abs() +
            (self.response - other.response).abs();
        if self.activation != other.activation { d += 1.0; }
        if self.aggregation != other.aggregation { d += 1.0; }
        d * config::COMPAT_WEIGHT_COEFFICIENT
    }
}

#[derive(Debug, Clone, Copy)]
pub (crate) struct ConnectionGene {
    id: usize,
    weight: f64,
    enabled: bool
}

impl ConnectionGene {
    pub fn new(id: usize, weight: f64, enabled: bool) -> Self {
        Self {
            id,
            weight, 
            enabled
        }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let mut d = (self.weight - other.weight).abs();
        if self.enabled != other.enabled { d += 1.0; }
        d * config::COMPAT_WEIGHT_COEFFICIENT
    }
}