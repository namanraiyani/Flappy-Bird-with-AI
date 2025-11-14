import numpy as np
import copy

class Agent:
    def __init__(self, input_count, layer_sizes, output_count):
        self.layer_structure = []
        previous_layer_size = input_count
        
        for size in layer_sizes:
            self.layer_structure.append((previous_layer_size, size))
            previous_layer_size = size
            
        self.layer_structure.append((previous_layer_size, output_count))
        
        rng = np.random.default_rng()
        self.synaptic_weights = [rng.normal(0, 0.5, size=shape) for shape in self.layer_structure]
        self.neuron_biases = [np.zeros((shape[1],), dtype=np.float32) for shape in self.layer_structure]
        
    def create_replica(self):
        replica = object.__new__(Agent)
        replica.layer_structure = copy.deepcopy(self.layer_structure)
        replica.synaptic_weights = [copy.deepcopy(w) for w in self.synaptic_weights]
        replica.neuron_biases = [copy.deepcopy(b) for b in self.neuron_biases]
        return replica
    
    def apply_genetic_mutation(self, weight_variance, bias_variance, mutation_rate):
        rng = np.random.default_rng()
        for i, (weights, biases) in enumerate(zip(self.synaptic_weights, self.neuron_biases)):
            if rng.random() < 0.9:
                gene_mask_w = (rng.random(weights.shape) < mutation_rate)
                weight_noise = rng.normal(0, weight_variance, size=weights.shape)
                self.synaptic_weights[i] = weights + (weight_noise * gene_mask_w)
                
                gene_mask_b = (rng.random(biases.shape) < mutation_rate)
                bias_noise = rng.normal(0, bias_variance, size=biases.shape)
                self.neuron_biases[i] = biases + (bias_noise * gene_mask_b)
    
    def compute_output(self, sensor_inputs):
        signal = sensor_inputs
        
        for (W, b) in list(zip(self.synaptic_weights, self.neuron_biases))[:-1]:
            signal = activation_tanh(signal @ W + b)
        
        final_output = signal @ self.synaptic_weights[-1] + self.neuron_biases[-1]
        return final_output
    
    def decide_action(self, environment_data):
        raw_decision = self.compute_output(environment_data)
        return 1 if raw_decision[0] > 0.0 else 0

def activation_tanh(value):
    return np.tanh(value).astype(np.float32)