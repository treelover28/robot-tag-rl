from tracemalloc import start
import numpy as np
from collections import deque
from math import sqrt

class NN_Layer:
    def __init__(self, num_input_nodes, num_output_nodes, activation_function, activation_function_derivative, learning_rate):
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.learning_rate = learning_rate
        
        # use He initialization to initialize layer's weight
        # where eah weight is a random number from a Gaussian distribution
        # with mean 0 and standard deviation of square-root(2/num_input_nodes)
        standard_deviation = sqrt(2.0/num_input_nodes)
        weights_unrolled = np.random.randn(self.num_input_nodes * self.num_output_nodes) * standard_deviation
        self.weights = np.reshape(weights_unrolled, (num_input_nodes, num_output_nodes))
        
    def forward_propagation(self, layer_input, store_values_for_backprop = True):
        # assume input is a column vector 
        layer_input_with_bias = np.append(np.ones((len(layer_input), 1)), layer_input, axis = 1)
        
        # (1 x previous layer_output_size) . (current_layer_input_size, current_layer_output_size)
        # where current_layer_input_size = previous_layer_output 
        unactivated_output =  np.dot(layer_input_with_bias, self.weights)
        layer_output = unactivated_output
        
        if self.activation_function != None:
            layer_output = self.activation_function(layer_output)

        if store_values_for_backprop:
            self.backward_input = layer_input_with_bias
            self.backward_output = unactivated_output
    
    def backward_propagation(self, gradient_from_subsequent_layer):
        if self.activation_function != None:
            adjustment = np.multiply(self.activation_function_derivative(self.backward_output), gradient_from_subsequent_layer)
            # update weight based on gradient
            gradient = np.dot(self.backward_input, adjustment)
            self.update_weight(gradient)
            delta = np.dot(adjustment, self.weights)
            return delta 
    
    def update_weight(self,gradient):
        self.weights += (self.learning_rate *gradient)

class DQN_Agent:
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size, start_epsilon, max_epsilon, activation_function, activation_function_derivative, learning_rate, discount_factor):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.start_epsilon = start_epsilon
        self.max_epsilon = max_epsilon
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # replay buffer to store experiences for experience replay
        self.replay_buffer = deque(maxlen=1000)

        self.layers = []
        # append input layer
        self.layers.append(NN_Layer(num_input_nodes = self.input_layer_size + 1, \
                                    num_output_nodes= self.hidden_layer_size, \
                                    activation_function=self.activation_function, \
                                    activation_function_derivative= self.activation_function_derivative, \
                                    learning_rate= self.learning_rate))
        # append hidden layers
        for i in range(self.num_hidden_layers):
            self.layers.append(NN_Layer(num_input_nodes = self.hidden_layer_size + 1, \
                                    num_output_nodes= self.hidden_layer_size, \
                                    activation_function=self.activation_function, \
                                    activation_function_derivative= self.activation_function_derivative, \
                                    learning_rate= self.learning_rate))
        # append output layer 
        self.layers.append(NN_Layer(num_input_nodes = self.hidden_layer_size + 1, \
                                    num_output_nodes= self.output_layer_size, \
                                    activation_function=self.None, \
                                    activation_function_derivative= None, \
                                    learning_rate= self.learning_rate))
    
    def get_action_q_values(self, state, remember_for_backprop):
        q_values = np.copy(state)
        # forward pass to get vector of action's q-values
        for layer in self.layers:
            q_values = layer.forward_propagation(q_values, remember_for_backprop)
        return q_values

    def get_policy(self,state, epsilon, action_list):
        # do a forward pass in Q-Network
        q_values = self.get_action_q_values(state, True)
        chosen_action = None
        if np.random.random() < epsilon:
            chosen_action = action_list[np.argmax(q_values)]
        else:
            chosen_action=  action_list[int(np.random.random() * len(action_list))]

        return chosen_action

    def remember_experience(self, is_terminal, current_state, action, new_state, reward):
        self.replay_buffer.append([is_terminal, current_state, action, new_state, reward])

    def experience_replay(self, mini_batch_size):
        if len(self.replay_buffer) > mini_batch_size:
            experience_indices = np.random.choice(len(self.replay_buffer), mini_batch_size, replace=False)
            for idx in experience_indices:
                is_terminal, current_state, action, new_state, reward = self.replay_buffer[idx]
                # get current q-values of actions for current state
                current_q_values = self.get_action_q_values(current_state, True)
                # get current Q-network's prediction of q-values for next state
                next_q_values = self.get_action_q_values(new_state, False)
                
                # get target value based on the reward from the reward function
                # this is the "true" q_values you wish your network to converge to
                target_q_values = current_q_values

                if is_terminal:
                    target_q_values[action] = reward # this reward should be a bad one!
                else:
                    target_q_values[action] = reward + self.discount_factor * np.max(next_q_values)
                # adjust the weights on the network
                self.adjust_network_weights(current_q_values, target_q_values)

    def adjust_network_weights(self,current_q_values, target_q_values):
        residuals = target_q_values - current_q_values
        for layer_idx in range(len(self.layers))[::-1]:
            layer = self.layers[layer_idx]
            residual = layer.backward_propagation(residual)
            


