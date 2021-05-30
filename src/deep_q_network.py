import numpy as np
import matplotlib.pyplot as plt 
import rospy
import cPickle as pickle
from collections import deque
from math import sqrt
# from robot_tag import random_walk_behavior


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
    def __init__(self, agent_type, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size, start_epsilon, max_epsilon, activation_function, activation_function_derivative, learning_rate, discount_factor, get_agent_state_function, agent_take_action_function, action_space):
        
        self.agent_type = agent_type
        self.agent_training_type = "DQN"
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
        self.get_agent_state_function = get_agent_state_function
        self.agent_take_action_function = agent_take_action_function
        self.action_space = action_space
        
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
    

    def get_action_q_values(self,remember_for_backprop):
        state = self.get_agent_state_function(self.agent_type)
        q_values = np.copy(state)
        
        # forward pass to get vector of action's q-values
        for layer in self.layers:
            q_values = layer.forward_propagation(q_values, remember_for_backprop)
        return q_values

    def get_policy(self, state, epsilon, verbose=False):
        # do a forward pass in Q-Network
        q_values = self.get_action_q_values(state, True)
        chosen_action = None
        if np.random.random() < epsilon:
            status = "Q-Network chose best action"
            chosen_action = self.action_space[np.argmax(q_values)]
        else:
            status = "Q-Network chose random action"
            chosen_action=  self.action_space[int(np.random.random() * len(self.action_space))]

        if verbose:
            rospy.loginfo(status)

        return chosen_action

    def follow_policy(self, time_to_apply_action = 0.33, evader_random_walk = False):
        current_state = self.get_agent_state_function(self.agent_type)
        
        # if self.agent_type == "pursuer": 
        #     translation_velocity, angular_velocity = self.get_policy(current_state, epsilon = 1.0, verbose= False)

        # else:
        #     if evader_random_walk:
        #         translation_velocity, angular_velocity = random_walk_behavior(robot_type="evader", robot_state=current_state)
        #     else:
        #         translation_velocity, angular_velocity = self.get_policy(current_state, epsilon = 1.0, verbose= False)
        translation_velocity, angular_velocity = self.get_policy(current_state, epsilon = 1.0, verbose= False)
        self.agent_take_action_function(self.agent_type, translation_velocity, angular_velocity)
        rospy.sleep(time_to_apply_action)

    def remember_experience(self, current_state, action, new_state, reward):
        self.replay_buffer.append([current_state, action, new_state, reward])

    def experience_replay(self, mini_batch_size):
        if len(self.replay_buffer) > mini_batch_size:
            experience_indices = np.random.choice(len(self.replay_buffer), mini_batch_size, replace=False)
            for idx in experience_indices:
                current_state, action, new_state, reward = self.replay_buffer[idx]
                # get current q-values of actions for current state
                current_q_values = self.get_action_q_values(current_state, True)
                # get current Q-network's prediction of q-values for next state
                next_q_values = self.get_action_q_values(new_state, False)
                
                # get target value based on the reward from the reward function
                # this is the "true" q_values you wish your network to converge to
                target_q_values = current_q_values

                # if is_terminal:
                #     target_q_values[action] = reward # this reward should be a bad one!
                # else:
                #     target_q_values[action] = reward + self.discount_factor * np.max(next_q_values)
                
                action_idx = self.action_space.index(action)
                target_q_values[action_idx] = reward + self.discount_factor * np.max(next_q_values)
                
                # adjust the weights on the network
                self.adjust_network_weights(current_q_values, target_q_values)

    def adjust_network_weights(self,current_q_values, target_q_values):
        residual = target_q_values - current_q_values
        for layer_idx in range(len(self.layers))[::-1]:
            layer = self.layers[layer_idx]
            residual = layer.backward_propagation(residual)

    def reward_function(self, state, verbose = True):
        #TODO
        pass
    
    def learn(self, epsilon, time_to_apply_action = 0.33):
        # Learn using Q-Learning
        # does one q-value update
        current_state = self.get_agent_state_function(self.agent_type)
            
        rospy.loginfo("Epsilon: {}".format(epsilon))
        
        # get action A from S using policy
        chosen_action = self.get_policy(current_state, epsilon= epsilon, verbose = True)
        translation_speed, turn_action = chosen_action
        
        # take action A and move player, this would change the player's state
        self.agent_take_action_function(self.agent_type, translation_speed, turn_action)
        
        # give the robot some time to apply action => proper state transition
        rospy.sleep(time_to_apply_action)
        
        # robot is now in new state S' 
        new_state = self.get_agent_state_function(self.agent_type)
        
        # robot now observes reward R(S') at this new state S'
        reward = self.reward_function(new_state)
        
        self.remember_experience(current_state, chosen_action, new_state, reward)
        self.experience_replay(mini_batch_size= 32)

        

        return reward, current_state, translation_speed, turn_action
    def save_agent(self, file_name):
        with open("{}".format(file_name), "w") as file_pointer:
            file_pointer.seek(0) # evader_rescue_thread.join()
            file_pointer.write(pickle.dumps(self))

