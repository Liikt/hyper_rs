use hashbrown::HashMap;

use crate::gene::{NodeGene, ConnectionGene};

pub struct Genome {
    innovation: usize,
    x_input: NodeGene,
    y_input: NodeGene,
    output: NodeGene,
    nodes: HashMap<usize, NodeGene>,
    connections: HashMap<usize, ConnectionGene>,
}