mod ngen {

    extern crate rand;

    use self::rand::{Rng,ThreadRng};
    use std::cmp::Ordering;
    use std::f64::INFINITY; 

    fn rand_clamped(rng: &mut ThreadRng) -> f64 {
        rng.gen::<f64>() - rng.gen::<f64>()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    #[derive(Clone,Copy,Debug)]
    pub struct NNetParams {
        chunk_size: usize,
        num_inputs: usize,
        num_layers: usize,
        num_neurons: usize,
        num_neurons_per_layer: usize,
        num_outputs: usize,
        num_weights: usize,
    }

    impl NNetParams {

        pub fn new(num_inputs: usize, num_layers: usize, num_neurons_per_layer: usize, num_outputs: usize) -> NNetParams {
            let chunk_size = num_inputs + num_outputs;
            let num_neurons = num_layers * num_neurons_per_layer + num_outputs;
            let num_weights = num_inputs * num_neurons_per_layer + (num_layers - 1) * num_neurons_per_layer * num_neurons_per_layer + num_neurons_per_layer * num_outputs;
            NNetParams{
                chunk_size,
                num_inputs,
                num_layers,
                num_neurons,
                num_neurons_per_layer,
                num_outputs,
                num_weights,
            }
        }
    }

    #[derive(Clone,Debug)]
    pub struct NNet {
        biases: Vec<f64>,
        cost: f64,
        fitness: f64,
        params: NNetParams,
        results: Vec<f64>,
        weights: Vec<f64>,
    }

    impl PartialEq for NNet {
        fn eq(&self, other: &NNet) -> bool {
            self.cost == other.cost
        }
    }

    impl PartialOrd for NNet {
        fn partial_cmp(&self, other: &NNet) -> Option<Ordering> {
            self.cost.partial_cmp(&other.cost)
        }
    }

    type CostFn = fn(&[f64],&[f64]) -> f64;

    impl NNet {

        fn new(params: NNetParams, rng: &mut ThreadRng) -> NNet {
            let biases = (0..params.num_neurons).map(|_| rand_clamped(rng)).collect::<Vec<_>>();
            let results = Vec::with_capacity(params.num_neurons);
            let weights = (0..params.num_weights).map(|_| rand_clamped(rng)).collect::<Vec<_>>();
            NNet{
                biases,
                cost: 0.0,
                fitness: 0.0,
                params,
                results,
                weights,
            }
        }

        pub fn eval(&mut self, inputs: &[f64], outputs: &[f64], cost_fn: CostFn) {
            // First layer
            let mut b = 0;
            let mut ws = 0;
            let mut we;
            for _ in 0..self.params.num_neurons_per_layer {
                we = ws + self.params.num_inputs;
                let res = self.weights[ws..we].iter().zip(inputs.iter()).fold(-self.biases[b], |acc, (w,x)| acc + w * x);
                let res = sigmoid(res);
                self.results.push(res);
                b += 1;
                ws += self.params.num_inputs;
            }
            // Middle layers
            let mut rs = 0;
            let mut re;
            for _ in 0..self.params.num_layers - 1 {
                re = rs + self.params.num_neurons_per_layer;
                for _ in 0..self.params.num_neurons_per_layer {
                    we = ws + self.params.num_neurons_per_layer;
                    let res = self.weights[ws..we].iter().zip(&self.results[rs..re]).fold(-self.biases[b], |acc, (w,x)| acc + w * x);
                    let res = sigmoid(res);
                    self.results.push(res);
                    b += 1;
                    ws += self.params.num_neurons_per_layer;
                }
                rs += self.params.num_neurons_per_layer;
            }
            // Last layer
            re = rs + self.params.num_neurons_per_layer;
            for _ in 0..self.params.num_outputs {
                we = ws + self.params.num_neurons_per_layer;
                let res = self.weights[ws..we].iter().zip(&self.results[rs..re]).fold(-self.biases[b], |acc, (w,x)| acc + w * x);
                let res = sigmoid(res);
                self.results.push(res);
                b += 1;
                ws += self.params.num_neurons_per_layer;
            }
            self.cost += cost_fn(&self.results[re..], outputs);
            self.results.clear();
        }
    }

    #[derive(Clone,Copy,Debug)]
    pub struct EnvParams {
        cost_thresh: f64,
        cross_rate: f64,
        iters: usize,
        max_perturb: f64,
        mut_rate: f64,
        num_copies: usize,
        num_holdovers: usize,
        num_new: usize,
        pop_size: usize,
    }

    impl EnvParams {
        // (num_copies * num_holdovers) % 2 == 0 
        // pop_size % 2 == 0
        pub fn new(cost_thresh: f64, cross_rate: f64, max_perturb: f64, mut_rate: f64, num_copies: usize, num_holdovers: usize, pop_size: usize) -> EnvParams {
            let num_new = pop_size - num_holdovers;
            let iters = (pop_size - (num_copies + 1) * num_holdovers) / 2;
            EnvParams{
                cost_thresh,
                cross_rate,
                iters,
                max_perturb,
                mut_rate,
                num_copies,
                num_holdovers,
                num_new,
                pop_size,
            }
        }
    }

    pub struct NGen {
        avg_fitness: f64,
        best_fitness: f64,
        cost_fn: CostFn,
        ngen_params: EnvParams,
        epochs: usize,
        new_weights: Vec<f64>,
        nnet_params: NNetParams,
        pop: Vec<NNet>,
        total_fitness: f64,
        worst_fitness: f64,
    }

    impl NGen {

        pub fn new(cost_fn: CostFn, ngen_params: EnvParams, nnet_params: NNetParams, rng: &mut ThreadRng) -> NGen {
            let pop = (0..ngen_params.pop_size).map(|_| NNet::new(nnet_params, rng)).collect::<Vec<_>>();
            let new_weights = vec![0.0; ngen_params.num_new * nnet_params.num_weights];
            NGen{
                avg_fitness: 0.0,
                best_fitness: 0.0,
                epochs: 0,
                cost_fn,
                new_weights,
                ngen_params,
                nnet_params,
                pop,
                total_fitness: 0.0,
                worst_fitness: 0.0,
            }
        }

        pub fn epoch(&mut self, data: &[f64], rng: &mut ThreadRng) -> Option<NNet> {
            if let Some(soln) = self.eval(data) {
                return Some(soln)
            }
            let mut w1 = 0;
            let mut w2;
            for i in 0..self.ngen_params.num_holdovers {
                for _ in 0..self.ngen_params.num_copies {
                    w2 = w1 + self.nnet_params.num_weights;
                    self.new_weights[w1..w2].copy_from_slice(&self.pop[i].weights);
                    w1 = w2;
                }
            }
            let mut we;
            for _ in 0..self.ngen_params.iters {
                w2 = w1 + self.nnet_params.num_weights;
                we = w2 + self.nnet_params.num_weights;
                let i = self.roulette(rng);
                let j = self.roulette(rng);
                self.new_weights[w1..w2].copy_from_slice(&self.pop[i].weights);
                self.new_weights[w2..we].copy_from_slice(&self.pop[j].weights);
                if i != j {
                    self.crossover(w1, w2, rng);
                }
                self.mutate(w1, w2, rng);
                w1 = we;
            }
            self.update();
            None
        }

        fn eval(&mut self, data: &[f64]) -> Option<NNet> {
            let num_chunks = data.len() as f64 / self.nnet_params.chunk_size as f64;
            self.best_fitness = 0.0;
            self.total_fitness = 0.0;
            self.worst_fitness = INFINITY;
            for nnet in self.pop.iter_mut() {
                nnet.cost = 0.0;
                for chunk in data.chunks(self.nnet_params.chunk_size) {
                    let inputs = &chunk[..self.nnet_params.num_inputs];
                    let outputs = &chunk[self.nnet_params.num_inputs..];
                    nnet.eval(inputs, outputs, self.cost_fn);
                }
                nnet.cost /= num_chunks;
                if nnet.cost < self.ngen_params.cost_thresh {
                    return Some(nnet.clone())
                } else {
                    nnet.fitness = 1.0 / nnet.cost;
                    if nnet.fitness > self.best_fitness {
                        self.best_fitness = nnet.fitness;
                    } else if nnet.fitness < self.worst_fitness {
                        self.worst_fitness = nnet.fitness;
                    }
                    self.total_fitness += nnet.fitness;
                }
            }
            self.avg_fitness = self.total_fitness / (self.ngen_params.pop_size as f64);
            None
        }

        fn update(&mut self) {
            self.pop.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let mut ws = 0;
            let mut we;
            for nnet in self.pop[self.ngen_params.num_holdovers..].iter_mut() {
                we = ws + self.nnet_params.num_weights;
                nnet.weights.copy_from_slice(&self.new_weights[ws..we]);
                ws += self.nnet_params.num_weights;
            }
            self.epochs += 1;
        }

        fn roulette(&self, rng: &mut ThreadRng) -> usize {
            let n = self.total_fitness * rng.gen::<f64>();
            let mut acc = 0.0;
            let mut i = 0;
            for nnet in self.pop.iter() {
                acc += nnet.fitness;
                if acc >= n {
                    break
                }
                i += 1;
            }
            i
        }

        fn mutate(&mut self, w1: usize, w2: usize, rng: &mut ThreadRng) {
            for i in 0..self.nnet_params.num_weights {
                if rng.gen::<f64>() < self.ngen_params.mut_rate {
                    self.new_weights[w1 + i] += rand_clamped(rng) * self.ngen_params.max_perturb;
                } 
                if rng.gen::<f64>() < self.ngen_params.mut_rate {
                    self.new_weights[w2 + i] += rand_clamped(rng) * self.ngen_params.max_perturb;
                } 
            }
        }

        fn crossover(&mut self, w1: usize, w2: usize, rng: &mut ThreadRng) -> usize {
            if rng.gen::<f64>() < self.ngen_params.cross_rate {
                let start : usize = rng.gen_range(0, self.nnet_params.num_weights);
                for i in start..self.nnet_params.num_weights {
                    let temp = self.new_weights[w1 + i];
                    self.new_weights[w1 + i] = self.new_weights[w2 + i];
                    self.new_weights[w2 + i] = temp;
                }
                start
            } else {
                self.nnet_params.num_weights
            }
        }
    }

    #[cfg(test)]
    mod test {

        use super::*;

        fn ngen_params() -> EnvParams {
            let cost_thresh = 0.1;
            let cross_rate = 0.7;
            let max_perturb = 0.3;
            let mut_rate = 0.3;
            let num_copies = 1;
            let num_holdovers = 4;
            let pop_size = 100;
            EnvParams::new(cost_thresh, cross_rate, max_perturb, mut_rate, num_copies, num_holdovers, pop_size)
        }

        fn nnet_params() -> NNetParams {
            let num_inputs = 4;
            let num_layers = 1;
            let num_neurons_per_layer = 6;
            let num_outputs = 1;
            NNetParams::new(num_inputs, num_layers, num_neurons_per_layer, num_outputs)
        }

        fn deep_nnet_params() -> NNetParams {
            let num_inputs = 4;
            let num_layers = 8;
            let num_neurons_per_layer = 6;
            let num_outputs = 1;
            NNetParams::new(num_inputs, num_layers, num_neurons_per_layer, num_outputs)
        }

        fn cost_fn(actual: &[f64], expected: &[f64]) -> f64 {
            actual.iter().zip(expected.iter()).fold(0.0, |acc, (a,b)| (a-b).abs() + acc) / (actual.len() as f64)
        }

        fn round_decimal_places(x: f64, dec_places: i32) -> f64 {
            let pow10 = (10.0 as f64).powi(dec_places);
            (x * pow10).round() / pow10
        }

        #[test]
        fn test_cost_fn() {
            let actual = [0.1, 0.2, 0.3];
            let expected = [0.4, 0.5, 0.6];
            let cost = cost_fn(&actual, &expected);
            let cost = round_decimal_places(cost, 1);
            assert_eq!(cost, 0.3);
        }

        #[test]
        fn test_nnet() {
            let params = nnet_params();
            let mut rng = rand::thread_rng();
            let mut nnet = NNet::new(params, &mut rng);
            let inputs = vec![0.1, 0.2, 0.3, 0.4];
            let mut b = 0;
            let mut w = 0;
            // Hidden layer
            let results = (0..params.num_neurons_per_layer).map(|_| {
                let result = sigmoid(inputs.iter().fold(-nnet.biases[b], |acc, input| {
                    let acc = acc + input * nnet.weights[w];
                    w += 1;
                    acc
                }));
                b += 1;
                result
            }).collect::<Vec<_>>();
            // Outputs
            let outputs = (0..params.num_outputs).map(|_| {
                let output = sigmoid(results.iter().fold(-nnet.biases[b], |acc, res| {
                    let acc = acc + res * nnet.weights[w];
                    w += 1;
                    acc
                }));
                b += 1;
                output
            }).collect::<Vec<_>>();
            nnet.eval(&inputs, &outputs, cost_fn);
            assert_eq!(round_decimal_places(nnet.cost, 4), 0.0000);
        }

        #[test]
        fn test_deep_nnet() {
            let params = deep_nnet_params();
            let mut rng = rand::thread_rng();
            let mut nnet = NNet::new(params, &mut rng);
            let inputs = vec![0.1, 0.2, 0.3, 0.4];
            let mut b = 0;
            let mut w = 0;
            // First layer
            let mut results = (0..params.num_neurons_per_layer).map(|_| {
                let result = sigmoid(inputs.iter().fold(-nnet.biases[b], |acc, input| {
                    let acc = acc + input * nnet.weights[w];
                    w += 1;
                    acc
                }));
                b += 1;
                result
            }).collect::<Vec<_>>();
            // Middle layers
            for _ in 0..params.num_layers-1 {
                results = (0..params.num_neurons_per_layer).map(|_| {
                    let result = sigmoid(results.iter().fold(-nnet.biases[b], |acc, res| {
                        let acc = acc + res * nnet.weights[w];
                        w += 1;
                        acc
                    }));
                    b += 1;
                    result
                }).collect::<Vec<_>>();
            }
            // Outputs
            let outputs = (0..params.num_outputs).map(|_| {
                let output = sigmoid(results.iter().fold(-nnet.biases[b], |acc, res| {
                    let acc = acc + res * nnet.weights[w];
                    w += 1;
                    acc
                }));
                b += 1;
                output
            }).collect::<Vec<_>>();
            nnet.eval(&inputs, &outputs, cost_fn);
            assert_eq!(round_decimal_places(nnet.cost, 4), 0.0000);
        }

        fn new_ngen_with_nnets(rng: &mut ThreadRng) -> NGen {
            let ngen_params = ngen_params();
            let nnet_params = nnet_params();
            NGen::new(cost_fn, ngen_params, nnet_params, rng)
        }

        fn new_ngen_with_deep_nnets(rng: &mut ThreadRng) -> NGen {
            let ngen_params = ngen_params();
            let nnet_params = deep_nnet_params();
            NGen::new(cost_fn, ngen_params, nnet_params, rng)
        }

        fn test_crossover(ngen: &mut NGen, rng: &mut ThreadRng) {
            let w1 = 0;
            let w2 = ngen.nnet_params.num_weights;
            let we = w2 + ngen.nnet_params.num_weights;
            ngen.new_weights[..w2].copy_from_slice(&ngen.pop[0].weights);
            ngen.new_weights[w2..we].copy_from_slice(&ngen.pop[1].weights);
            let orig_weights1 = ngen.pop[0].weights.clone();
            let orig_weights2 = ngen.pop[1].weights.clone();
            let start = ngen.crossover(w1, w2, rng);
            if start == ngen.nnet_params.num_weights {
                assert_eq!(&ngen.new_weights[..w2], orig_weights1.as_slice());
                assert_eq!(&ngen.new_weights[w2..we], orig_weights2.as_slice());
            } else {
                let mut new_weights1 = Vec::with_capacity(ngen.nnet_params.num_weights);
                let mut new_weights2 = Vec::with_capacity(ngen.nnet_params.num_weights);
                new_weights1.extend_from_slice(&orig_weights1[..start]);
                new_weights1.extend_from_slice(&orig_weights2[start..]);
                new_weights2.extend_from_slice(&orig_weights2[..start]);
                new_weights2.extend_from_slice(&orig_weights1[start..]);
                assert_eq!(&ngen.new_weights[..w2], new_weights1.as_slice());
                assert_eq!(&ngen.new_weights[w2..we], new_weights2.as_slice());
            }
        }

        fn new_data(params: NNetParams, num_data: usize, rng: &mut ThreadRng) -> Vec<f64> {
            let mut data = Vec::with_capacity(num_data * (params.num_inputs + params.num_outputs));
            for _ in 0..num_data {
                let inputs = (0..params.num_inputs).map(|_| rng.gen::<f64>()).collect::<Vec<_>>();
                let output = inputs[0] + inputs[1].powi(2) - inputs[2].powi(3) + inputs[3].powi(4);
                data.extend(inputs);
                data.push(output);
            }
            data
        }

        fn rand_data(params: NNetParams, num_data: usize, rng: &mut ThreadRng) -> Vec<f64> {
            (0..num_data * (params.num_inputs + params.num_outputs)).map(|_| rand_clamped(rng)).collect::<Vec<_>>()
        }

        fn test_epoch(ngen: &mut NGen, rng: &mut ThreadRng) {
            let data = rand_data(ngen.nnet_params, 100, rng);
            ngen.epoch(&data, rng);
            let best = ngen.best_fitness;
            assert_eq!(ngen.epochs, 1);
            assert!(ngen.best_fitness >= best);
        }

        fn test_run(ngen: &mut NGen, rng: &mut ThreadRng) {
            let data = new_data(ngen.nnet_params, 10, rng);
            let mut soln = None;
            while soln.is_none() {
                soln = ngen.epoch(&data, rng);
                println!("Best fitness: {}, Worst fitness: {}, Total fitness: {}", ngen.best_fitness, ngen.worst_fitness, ngen.total_fitness);
            }
            let mut soln = soln.unwrap();
            soln.cost = 0.0;
            soln.eval(&[0.1, 0.2, 0.3, 0.4], &[0.1386], cost_fn);
            assert!(soln.cost <= ngen.ngen_params.cost_thresh);
            println!("Ran for {} epochs", ngen.epochs);
            println!("Cost: {}", soln.cost);
        }

        // #[test]
        fn test_ngen_with_nnets() {
            let mut rng = rand::thread_rng();
            let mut ngen = new_ngen_with_nnets(&mut rng);
            test_crossover(&mut ngen, &mut rng);
            test_epoch(&mut ngen, &mut rng);
            test_run(&mut ngen, &mut rng);
        }

        // #[test]
        fn test_ngen_with_deep_nnets() {
            let mut rng = rand::thread_rng();
            let mut ngen = new_ngen_with_deep_nnets(&mut rng);
            test_crossover(&mut ngen, &mut rng);
            test_epoch(&mut ngen, &mut rng);
            test_run(&mut ngen, &mut rng);
        }
    }
}