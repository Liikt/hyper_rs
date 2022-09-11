use rand::{distributions::{Distribution, Standard}, Rng};

const LAMBDA: f64 = 1.0507009873554805;
const ALPHA: f64  = 1.6732632423543772;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivationFunction {
    Sigmoid,
    Tanh,
    Sin,
    Gauss,
    Relu,
    Elu,
    Lelu,
    Selu,
    SoftPlus,
    Identity,
    Clamped,
    Inv,
    Log,
    Exp,
    Abs,
    Hat,
    Square,
    Cube
}

impl ActivationFunction {
    pub fn activate(&self, val: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                let z = (val * 5.0).clamp(-60.0, 60.0);
                1.0 / (1.0 + (-z).exp())
            }
            Self::Tanh => (val * 2.5).clamp(-60.0, 60.0).tanh(),
            Self::Sin => (val * 5.0).clamp(-60.0, 60.0).sin(),
            Self::Gauss => -5.0 * val.clamp(-3.4, 3.4).powi(2),
            Self::Relu => if val > 0.0 { val } else { 0.0 } 
            Self::Elu => if val > 0.0 { val } else { val.exp_m1() }
            Self::Lelu => if val > 0.0 { val } else { 0.005 * val }
            Self::Selu => {
                if val > 0.0 { 
                    LAMBDA * val 
                } else { 
                    LAMBDA * ALPHA * val.exp_m1() 
                }
            }
            Self::SoftPlus => {
                let z = (val * 5.0).clamp(-60.0, 60.0);
                0.2 * (1.0 + z.exp()).log10()
            }
            Self::Identity => val,
            Self::Clamped => val.clamp(-1.0, 1.0),
            Self::Inv => if val == 0.0 { 0.0 } else { 1.0 / val },
            Self::Log => val.max(1e-7).log10(),
            Self::Exp => val.clamp(-60.0, 60.0).exp(),
            Self::Abs => val.abs(),
            Self::Hat => (1.0 - val.abs()).max(0.0),
            Self::Square => val.powi(2),
            Self::Cube => val.powi(3)
        }
    }
}

impl Distribution<ActivationFunction> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ActivationFunction {
        match rng.gen_range(0..=17) {
                0  => ActivationFunction::Sigmoid,
                1  => ActivationFunction::Tanh,
                2  => ActivationFunction::Sin,
                3  => ActivationFunction::Gauss,
                4  => ActivationFunction::Relu,
                5  => ActivationFunction::Elu,
                6  => ActivationFunction::Lelu,
                7  => ActivationFunction::Selu,
                8  => ActivationFunction::SoftPlus,
                9  => ActivationFunction::Identity,
                10 => ActivationFunction::Clamped,
                11 => ActivationFunction::Inv,
                12 => ActivationFunction::Log,
                13 => ActivationFunction::Exp,
                14 => ActivationFunction::Abs,
                15 => ActivationFunction::Hat,
                16 => ActivationFunction::Square,
                _ => ActivationFunction::Cube,
        }
    }
}