use rand::{distributions::{Distribution, Standard}, Rng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggregationFunction {
    Product,
    Sum,
    Max,
    Min, 
    MaxAbs,
    Mean
}

impl AggregationFunction {
    pub fn aggregate(&self, vals: &[f64]) -> f64 {
        match self {
            Self::Product => vals.into_iter().fold(1.0, |acc, x| acc * x),
            Self::Sum => vals.into_iter().sum(),
            Self::Max => {
                if vals.is_empty() { return 0.0; }
                let mut x = vals[0];
                for &y in vals { if y > x { x = y; } }
                x
            }
            Self::Min => {
                if vals.is_empty() { return 0.0; }
                let mut x = vals[0];
                for &y in vals { if y < x { x = y; } }
                x
            }
            Self::MaxAbs => {
                if vals.is_empty() { return 0.0; }
                let mut x = vals[0];
                for &y in vals { if y.abs() > x { x = y; } }
                x
            }
            Self::Mean => vals.into_iter().sum::<f64>() / (vals.len() as f64),
        }
    }
}

impl Distribution<AggregationFunction> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> AggregationFunction {
        match rng.gen_range(0..=5) {
            0 => AggregationFunction::Product,
            1 => AggregationFunction::Sum,
            2 => AggregationFunction::Max,
            3 => AggregationFunction::Min,
            4 => AggregationFunction::MaxAbs,
            _ => AggregationFunction::Mean
        }
    }
}