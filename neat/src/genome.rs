use hashbrown::{HashMap, HashSet};
use rand::{random, thread_rng, seq::IteratorRandom};

use crate::{gene::{NodeGene, ConnectionGene}, config};

pub struct Genome {
    id:         usize,
    innovation: usize,
    fitness:    f64,

    nodes:       HashMap<usize, NodeGene>,
    connections: HashMap<usize, ConnectionGene>,
}

impl Genome {
    pub fn new(id: usize) -> Self {
        let mut ret = Genome {
            id,
            innovation: 2,
            fitness: 0.0,
            nodes: HashMap::new(),
            connections: HashMap::new()
        };

        ret.connections.insert(0, ConnectionGene::new(0, 0, 4, 1.0, true));
        ret.connections.insert(1, ConnectionGene::new(1, 1, 4, 1.0, true));
        ret.connections.insert(2, ConnectionGene::new(2, 2, 4, 1.0, true));
        ret.connections.insert(3, ConnectionGene::new(3, 3, 4, 1.0, true));

        ret.nodes.insert(0, NodeGene::new(0));
        (*ret.nodes.get_mut(&0).unwrap()).dst_connections.insert(0);
        
        ret.nodes.insert(1, NodeGene::new(1));
        (*ret.nodes.get_mut(&1).unwrap()).dst_connections.insert(1);
        
        ret.nodes.insert(2, NodeGene::new(2));
        (*ret.nodes.get_mut(&2).unwrap()).dst_connections.insert(2);
        
        ret.nodes.insert(3, NodeGene::new(3));
        (*ret.nodes.get_mut(&3).unwrap()).dst_connections.insert(3);

        ret.nodes.insert(4, NodeGene::new(4));
        (*ret.nodes.get_mut(&4).unwrap()).src_connections.insert(0);
        (*ret.nodes.get_mut(&4).unwrap()).src_connections.insert(1);
        (*ret.nodes.get_mut(&4).unwrap()).src_connections.insert(2);
        (*ret.nodes.get_mut(&4).unwrap()).src_connections.insert(3);


        ret
    }

    fn get_new_node_key(&self) -> usize {
        let mut r = random();
        while self.nodes.contains_key(&r) {
            r = random();
        }
        r
    }

    #[inline]
    fn get_new_conn_key(&mut self) -> usize {
        self.innovation += 1;
        self.innovation
    }

    fn add_connection(&mut self, src: usize, dst: usize, weight: f64, 
            enabled: bool) {
        let new_id = self.get_new_conn_key();
        self.connections.insert(new_id, ConnectionGene::new(
            new_id, src, dst, weight, enabled));
        (*self.nodes.get_mut(&src).unwrap()).dst_connections.insert(new_id);
        (*self.nodes.get_mut(&dst).unwrap()).src_connections.insert(new_id);
    }

    pub fn crossover(&mut self, genome1: &Self, genome2: &Self) {
        let (parent1, parent2) = if genome1.fitness > genome2.fitness { 
            (genome1, genome2) 
        } else { 
            (genome2, genome1)
        };

        for (k, conn1) in parent1.connections.iter() {
            match parent2.connections.get(&k) {
                Some(conn2) => {
                    self.connections.insert(*k, conn1.crossover(conn2));
                }
                None => { self.connections.insert(*k, *conn1); }
            }
        }

        for (k, node1) in parent1.nodes.iter() {
            assert!(self.nodes.get(k).is_none());
            match parent2.nodes.get(&k) {
                Some(node2) => {
                    self.nodes.insert(*k, node1.crossover(node2));
                }
                None => { self.nodes.insert(*k, node1.clone()); }
            }
        }
    }

    fn creates_circles(&self, src: usize, dst: usize) -> bool {
        if src == dst { return true; }
        let mut visited = HashSet::new();
        visited.insert(dst);
        loop {
            let mut num_added = 0;
            for conn in self.connections.values() {
                if visited.contains(&conn.get_src()) && 
                        !visited.contains(&conn.get_dst()) {
                    if conn.get_dst() == src {
                        return true;
                    }
                    visited.insert(conn.get_dst());
                    num_added += 1;
                }
            }
            if num_added == 0 {
                return true;
            }
        }
    }

    fn mutate_add_node(&mut self) {
        let mut rng = thread_rng();
        let new_node_id = self.get_new_node_key();
        let new_node = NodeGene::new(new_node_id);
        self.nodes.insert(new_node_id, new_node);

        let conn = self.connections.values_mut().choose(&mut rng).unwrap();
        (*conn).disable();
        let src = conn.get_src();
        let dst = conn.get_dst();
        let weight = conn.get_weight();

        self.add_connection(src, new_node_id, 1.0, true);
        self.add_connection(new_node_id, dst, weight, true);
    }

    fn mutate_add_conn(&mut self) {
        let mut rng = thread_rng();
        let mut out_node = self.nodes.values().choose(&mut rng).unwrap();
        while out_node.get_id() < 4 {
            out_node = self.nodes.values().choose(&mut rng).unwrap();
        }

        let mut in_node = self.nodes.values().choose(&mut rng).unwrap();
        while out_node.get_id() == 4 && in_node.get_id() == 4 {
            in_node = self.nodes.values().choose(&mut rng).unwrap();
        }

        // return if this connection already exists
        if in_node.dst_connections.iter().any(|x| {
            self.connections.get(x).unwrap().get_dst() == out_node.get_id()
        }) {
            return;
        }

        if self.creates_circles(in_node.get_id(), out_node.get_id()) {
            return;
        }

        let min = config::WEIGHT_MIN_VALUE;
        let max = config::WEIGHT_MAX_VALUE;
        let weight = (random::<f64>() % (min + max)) - min;
        self.add_connection(in_node.get_id(), out_node.get_id(), weight, true);
    }

    fn mutate_del_node(&mut self) {
        let mut rng = thread_rng();
        let node = self.nodes.keys().filter(|&&x| x > 4).choose(&mut rng);
        if node.is_none() {
            return;
        }
        let node = *node.unwrap();
    }

    fn mutate_del_conn(&mut self) {}

    pub fn mutate(&mut self) {
        if config::SINGLE_MUTATION {
            let div = (
                config::NODE_ADD_PROB + config::NODE_DEL_PROB +
                config::CONN_ADD_PROB + config::CONN_DEL_PROB
            ).max(1.0);

            let node_add_prob = config::NODE_ADD_PROB;
            let node_del_prob = node_add_prob + config::NODE_DEL_PROB;
            let conn_add_prob = node_del_prob + config::CONN_ADD_PROB;
            let conn_del_prob = conn_add_prob + config::CONN_DEL_PROB;

            let r = random::<f64>() % 1.0;
            if r < node_add_prob/div {
                self.mutate_add_node();
            } else if r < node_del_prob/div {
                self.mutate_del_node();
            } else if r < conn_add_prob/div {
                self.mutate_add_conn();
            } else if r < conn_del_prob/div {
                self.mutate_del_conn();
            }
        } else {
            if random::<f64>() % 1.0 < config::NODE_ADD_PROB {
                self.mutate_add_node();
            }
            if random::<f64>() % 1.0 < config::NODE_DEL_PROB {
                self.mutate_del_node();
            }
            if random::<f64>() % 1.0 < config::CONN_ADD_PROB {
                self.mutate_add_conn();
            }
            if random::<f64>() % 1.0 < config::CONN_DEL_PROB {
                self.mutate_del_conn();
            }
        }

        for conn in self.connections.values_mut() {
            (*conn).mutate();
        }

        for node in self.nodes.values_mut() {
            (*node).mutate();
        }
    }
}