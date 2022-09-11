pub const FITNESS_THRESH: f64 = 3.9;
pub const POP_SIZE: usize     = 150;

pub const COMPAT_DISJOINT_COEFFICIENT: f64 = 1.0;
pub const COMPAT_WEIGHT_COEFFICIENT: f64   = 1.0;

pub const ACTIVATION_MUT_PROB: f64  = 0.2;
pub const AGGREGATION_MUT_PROB: f64 = 0.2;

pub const BIAS_MUTATE_POWER: f64 = 0.5;
pub const BIAS_MUTATE_RATE: f64  = 0.7;
pub const BIAS_REPLACE_RATE: f64 = 0.1;
pub const BIAS_MAX_VALUE: f64    = 30.0;
pub const BIAS_MIN_VALUE: f64    = -30.0;

pub const CONN_ADD_PROB: f64 = 0.2;
pub const CONN_DEL_PROB: f64 = 0.2;
pub const NODE_ADD_PROB: f64 = 0.1;
pub const NODE_DEL_PROB: f64 = 0.1;

pub const ENABLE_PROB: f64 = 0.02;

pub const WEIGHT_MUTATE_POWER: f64 = 0.5;
pub const WEIGHT_MUTATE_RATE: f64  = 0.8;
pub const WEIGHT_REPLACE_RATE: f64 = 0.1;
pub const WEIGHT_MAX_VALUE: f64    = 30.0;
pub const WEIGHT_MIN_VALUE: f64    = -30.0;

pub const RESPONSE_REPLACE_RATE: f64 = 0.1;
pub const RESPONSE_MUTATE_RATE: f64  = 0.1;
pub const RESPONSE_MUTATE_POWER: f64 = 0.1;
pub const RESPONSE_MAX_VALUE: f64    = 30.0;
pub const RESPONSE_MIN_VALUE: f64    = -30.0;

pub const COMPATIBILITY_THRESH: f64 = 2.0;

pub const MAX_STAGNATION: f64 = 15.0;

pub const ELITISM: f64         = 2.0;
pub const SURVIVAL_THRESH: f64 = 0.2;

pub const SINGLE_MUTATION: bool = true;