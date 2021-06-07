

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt 
import rospy
import cPickle as pickle
from collections import deque
from math import sqrt, isnan
from base_agent import Base_Agent
from activation_functions import sigmoid, relu, relu_derivative
import os


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
        self.weights = np.random.randn(num_input_nodes,num_output_nodes) * standard_deviation
        # self.weights = np.reshape(weights_unrolled, (num_input_nodes, num_output_nodes))
        
    def forward_propagation(self, layer_input, store_values_for_backprop = True):
        # assume input is a column vector 
        layer_input = np.reshape(layer_input, (1, self.num_input_nodes - 1))
        bias_term = np.ones((1, 1))
        # add bias term to output
        layer_input_with_bias = np.append(bias_term, layer_input, axis = 1)
        # (1 x previous layer_output_size) . (current_layer_input_size, current_layer_output_size)
        # where current_layer_input_size = previous_layer_output 
        unactivated_output =  np.dot(layer_input_with_bias, self.weights)
        layer_output = unactivated_output
        
        if self.activation_function != None:
            layer_output = self.activation_function(layer_output)

        if store_values_for_backprop:
            self.backward_input = np.copy(layer_input_with_bias)
            self.backward_output = np.copy(unactivated_output)
        
        return layer_output
    
    def backward_propagation(self, delta_from_subsequent_layer):
        delta = delta_from_subsequent_layer
        
        if self.activation_function != None:
            # needs to multiply delta from subsequent layer with the derivative of activation function on the current layer's output
            # chain rule
            delta = np.multiply(self.activation_function_derivative(self.backward_output), delta_from_subsequent_layer)
        
        # actual gradient of loss function with respect to the weights
        gradient = np.dot(self.backward_input.T, delta)
        
        # perform gradient clipping to prevent exploding gradient issue:
        gradient_norm = np.linalg.norm(gradient)
        gradient_threshold = 1.0
        if gradient_norm > gradient_threshold:
            gradient = np.multiply(gradient_threshold, (gradient/gradient_norm))
        
        # perform one gradient descent step to update weights in network
        self.gradient_descent_step(gradient)
        
        # pass back delta to layers before
        delta = np.dot(delta, self.weights.T)[0][:-1]
        
        return delta 
    
    def gradient_descent_step(self,gradient):
        self.weights += (self.learning_rate *gradient)

    

class DQN_Agent(Base_Agent):
    def __init__(self, agent_type, input_layer_size, output_layer_size, hidden_layers, action_space, activation_function, activation_function_derivative, num_steps_to_update_network, batch_size, learning_rate, discount_factor, get_agent_state_function, agent_take_action_function, get_game_information): 
        # call parent's constructor
        super(DQN_Agent, self).__init__(agent_type, learning_rate, discount_factor, get_agent_state_function, agent_take_action_function, get_game_information)
        
        self.agent_algorithm = "DQN"
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.action_space = action_space
        self.batch_size = batch_size
        
     
        # replay buffer to store experiences for experience replay
        self.replay_buffer = deque(maxlen=1000)

        # layers of the Deep Q-Network
        self.network_layers = []
        # append input layer
        self.network_layers.append(NN_Layer(num_input_nodes = self.input_layer_size + 1, \
                                    num_output_nodes= self.hidden_layers[0], \
                                    activation_function=self.activation_function, \
                                    activation_function_derivative= self.activation_function_derivative, \
                                    learning_rate= self.learning_rate))
        
        # rospy.loginfo("Initial first layer's weight: {}".format(self.network_layers[0].weights))
        
        # append hidden layers
        for i in range(len(self.hidden_layers) - 1):
            self.network_layers.append(NN_Layer(num_input_nodes = self.hidden_layers[i] + 1, \
                                    num_output_nodes= self.hidden_layers[i + 1], \
                                    activation_function=self.activation_function, \
                                    activation_function_derivative= self.activation_function_derivative, \
                                    learning_rate= self.learning_rate))
        # append output layer 
        self.network_layers.append(NN_Layer(num_input_nodes = self.hidden_layers[-1] + 1, \
                                    num_output_nodes= self.output_layer_size, \
                                    activation_function=None, \
                                    activation_function_derivative= None, \
                                    learning_rate= self.learning_rate))

        # this target network holds the frozen weights of the policy network
        # normally, we do the first pass through the network to approximate the Q-value of state s, => Q(s,a)
        # and then we do a second pass through the network to calculate the Q-value for the subsequent state s'.
        # from this second pass, we obtain the maximum Q-value amongst all action that the agent could take when it transitions
        # to the next state, then we can plug it back to the Bellman equation to calculate the target Q-value
        # for the action chosen for the first pass with state s.
        # Q*(s,a) = E[R(s') + discount_factor x Q(s',a')]
        # Our goal is to have our network's approximation of Q(s,a) to be close to the target Q*(s,a)

        # However, there is an issue with this process => since we use the same network to calculate Q(s,a) and Q(s',a'),
        # As the weights update such that Q(s,a) move closer to Q*(s,a), the target Q*(s,a) would also be moving in the same direction!
        # => like a dog chasing its own tail! => causes instability in learning
        
        # Thus we need to freeze the parameters of our current network into a clone target network
        # Now, we do the first pass through the current policy network to calculate Q(s,a; theta_current)
        # Then we do the second pass through the TARGET network with the frozen weights to calculate Q(s',a'; theta_frozen)
        # Again, we could plug this into Bellman equation => target Q*(s,a) which DOES NOT MOVE
        # Our network has a FIXED target to try to approximate now => remove in the instability in learning
        # Every C time steps, we update the target network to our latest network; theta_frozen = theta_current
        self.target_network = None
        self.num_steps_to_update_network = num_steps_to_update_network
        self.num_time_steps_without_update = 0
    
    
    def _get_action_q_values(self,state, remember_for_backprop):
        q_values = np.copy(state)
        # forward pass through Q-network to get vector of action's q-values
        for layer in self.network_layers:
            q_values = layer.forward_propagation(q_values, remember_for_backprop)
        return q_values

    def get_policy(self, state, epsilon, verbose=False):
        # do a forward pass in Q-Network
        q_values = self._get_action_q_values(state, True)
            
        chosen_action = None
        if np.random.random() < epsilon:
            chosen_action = self.action_space[np.argmax(q_values)]
            status = "Q-Network chose best action {}".format(chosen_action)
        else:
            chosen_action=  self.action_space[int(np.random.random() * len(self.action_space))]
            status = "Q-Network chose random action {}".format(chosen_action)

        if verbose:
            rospy.loginfo("DQN-Agent's current state is {}".format(state))
            rospy.loginfo(status)

        return chosen_action


    def follow_policy(self, time_to_apply_action = 0.33, verbose = True):
        # get agent's state in environment
        current_state = self.get_agent_state_function(self.agent_type, "continuous")

        # get optimal policy based on state
        translation_velocity, angular_velocity = self.get_policy(current_state, epsilon = 0.95, verbose= True)
        # make agent take action
        self.agent_take_action_function(self.agent_type, translation_velocity, angular_velocity, time_to_apply_action)


    def _memorize_experience(self, current_state, action, new_state, reward, is_terminal):
        self.replay_buffer.append(np.array([current_state, action, new_state, reward, is_terminal]))

    def _experience_replay(self, mini_batch_size, verbose = True):
        if len(self.replay_buffer) > mini_batch_size:

            if self.num_time_steps_without_update % self.num_steps_to_update_network == 0:
                
                # clone current network to target network every C time steps
                # for details regarding why we are using the target network, see __init__() function
                rospy.loginfo("UPDATED TARGET NETWORK TO CURRENT NETWORK")
                
                # set target network to none first to prevent recursive memory copying
                self.target_network = None 
                self.target_network = deepcopy(self)
              
                # reset time step
                self.num_time_steps_without_update = 0
            
            # rospy.loginfo("Target network's weight {}".format(self.target_network.network_layers[0].weights[0]))
            
            
            # uniformly sample random experiences in the past to decorrelate learning with regards to experiences to follow one another
            experience_indices = np.random.choice(len(self.replay_buffer), mini_batch_size, replace=False)
            accumulated_loss = 0.0
            accumulated_current_q = 0.0
            accumulated_target_q = 0.0

            for idx in experience_indices:
                current_state, action, new_state, reward, is_terminal = self.replay_buffer[idx]
                
                # get current network's approximation of q-values of actions Q(s,a; theta_current) for current state s
                current_q_values = self._get_action_q_values(current_state, True)

                # get target network's approximation of q-values of action Q(s',a'; theta_frozen) for next state s'
                next_q_values = self.target_network._get_action_q_values(new_state, False)
                
                # The target Q-value Q*(s,a) could be found via the Bellman Equation
                # Q*(s,a) = E[R(s') + discount_factor * argmax_a' Q(s',a')]
                # Thus, you wish to update the weight as such Q(s,a; theta_current) closely approximate the target Q*(s,a)
                target_q_values = np.copy(current_q_values)
                # get index of action
                action_idx = self.action_space.index(action)
                
                # update q_value of action Q(s,a) by Bellman to obtain Q*(s,a)
                if is_terminal:
                    # if it is terminal state, there is no more future states to accumulate discounted reward for
                    target_q_values[0][action_idx] = reward
                else:
                    target_q_values[0][action_idx] = reward + self.discount_factor * np.max(next_q_values)
                
                # adjust the weights on the current network
                self._optimize_network(current_q_values, target_q_values)

                # these are done just for diagnostic/reporting purposes
                accumulated_current_q += (np.sum(current_q_values))/len(current_q_values[0])
                accumulated_target_q += (np.sum(target_q_values))/len(target_q_values[0])
                accumulated_loss += ((np.sum(target_q_values - current_q_values))**2)
                
                # terminate program if gradient exploded, which hopefully will never happen
                for q_val in current_q_values[0]:
                    if isnan(q_val):
                        rospy.loginfo("GRADIENT EXPLODED")
                        exit(0)
            
            self.num_time_steps_without_update += 1
            
            if verbose:
                rospy.loginfo("Root Mean Squared Error is {}".format((accumulated_loss/mini_batch_size)**0.5))
                rospy.loginfo("Average current Q is {}".format(accumulated_current_q/mini_batch_size))
                rospy.loginfo("Average target Q is {}".format(accumulated_target_q/mini_batch_size))
                
            return (accumulated_loss/mini_batch_size)**0.5, accumulated_current_q/mini_batch_size

    def _optimize_network(self,current_q_values, target_q_values):
        
        residual =  (target_q_values - current_q_values)
        # perform backpropagation to adjust network's weights
        for layer_idx in range(len(self.network_layers))[::-1]:
            layer = self.network_layers[layer_idx]
            residual = layer.backward_propagation(residual)
            

    
    def reward_function(self, state, verbose = True):
        # get in-game information
        pursuer_radius = self.get_game_information("pursuer_radius")
        evader_radius = self.get_game_information("evader_radius")
        safe_distance_from_obstacle = self.get_game_information("safe_distance_from_obstacle")
        pursuer_stuck = self.get_game_information("pursuer_stuck")
        evader_stuck = self.get_game_information("evader_stuck")
        pursuer_was_stuck_but_rescued = self.get_game_information("pursuer_was_stuck_but_rescued")
        evader_was_stuck_but_rescued = self.get_game_information("evader_was_stuck_but_rescued")
        pursuer_min_distance_to_obstacle = self.get_game_information("pursuer_min_distance_to_obstacle")
        evader_min_distance_to_obstacle = self.get_game_information("evader_min_distance_to_obstacle")
        pursuer_min_distance_to_obstacle_direction = self.get_game_information("pursuer_min_distance_to_obstacle_direction")
        game_timeout = self.get_game_information("game_timeout")
        num_times_evader_stuck_in_episode = self.get_game_information("num_times_evader_stuck_in_episode")
        distance_between_players = self.get_game_information("distance_between_players")


        is_terminal_state = False
        
        if self.agent_type == "pursuer":
            # if the pursuer gets stuck, it loses that game -> negative reward
            # the negative reward is also based on how badly it lost that round

            # the distance is between player calculated from the positions is the distance from one's robot's center to another robot's center
            # while distance gathered from the pursuer's LIDAR is from the pursuer's center to the evader's nearest SIDE 
            # thus, we need to adjust this DISTANCE_BETWEEN_PLAYERS by the evader's radius to better compare the two 
            true_distance_between_player = (distance_between_players - evader_radius)
            true_safe_distance_from_obstacle = (pursuer_radius + safe_distance_from_obstacle)

            # rospy.loginfo("PURSUER_MIN_DISTANCE: {}\nTRUE_DISTANCE_BETWEEN_PLAYERS: {}\nTRUE_SAFE_DISTANCE_FROM_OBSTACLE: {}".format(PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_DISTANCE_BETWEEN_PLAYERS ,TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
            if pursuer_stuck or pursuer_was_stuck_but_rescued:
                # rospy.loginfo("STUCK!")
                state_description = "STUCK!"
                reward = -30 * sigmoid(true_distance_between_player)
                is_terminal_state = True
            elif state["Opponent Position"] == "Tagged":
                # rospy.loginfo("TAGGED!")
                state_description = "TAGGED!"
                reward = 30 
                is_terminal_state = True

            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] in ["Close", "OK"] and state["Opponent Position"] in ["Front", "Close Front"]:
                
                state_description = "Obstacle on both sides, but there is opening in front and the target is in front nearby"
                reward = -sigmoid(1/pursuer_min_distance_to_obstacle) + 2*sigmoid(1/true_distance_between_player)
            
            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] == "Far" and state["Opponent Position"] == "Front":
                
                state_description = "Obstacle on both sides, but there is opening in front and the target is in front far away"
                reward = -sigmoid(1/pursuer_min_distance_to_obstacle) + sigmoid(1/true_distance_between_player) 
            
            # there are obstacle on BOTH sides but there is an opening in front, and opponent is also in front
            elif (state["Upper Left"] in ["Close","Too Close"] or state["Upper Right"] in ["Close","Too Close"])\
                and (state["Upper Right"] in ["Close", "Too Close"] or state["Lower Right"] in ["Close", "Too Close"])\
                and state["Front"] != "Close":

                state_description = "Obstacle on both sides, but there is opening in front but opponent is not in front"
                reward = -sigmoid(1/pursuer_min_distance_to_obstacle) - sigmoid(true_distance_between_player) 
            
            # if there are obstacles nearby ON ONE SIDE(that is not the evader), and the evader is far away, promote obstacles avoidance behavior
            elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left")  or \
                    (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right") or \
                    (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Left") or \
                    (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] != "Close Right"))
                ) and true_distance_between_player > true_safe_distance_from_obstacle\
                    and pursuer_min_distance_to_obstacle_direction != "Front": 

                state_description = "Obstacle is a lot nearer on the sides compared to evader. Prioritize obstacle avoidance"
                # extra punishment depending on how far the evader is and how close the pursuer is to an obstacle
                reward = -0.5 - sigmoid(true_distance_between_player) - sigmoid(1/pursuer_min_distance_to_obstacle)
            
            # there is an obstacle in front that is not the opponent
            elif (state["Front"] == "Close" or (pursuer_min_distance_to_obstacle_direction == "Front" and pursuer_min_distance_to_obstacle <= true_safe_distance_from_obstacle))\
                and state["Opponent Position"] not in  ["Front", "Close Front"]:
                
                state_description = "Obstacle directly infront that is not the opponent. Prioritize obstacle avoidance"
                reward = -0.25 - sigmoid(true_distance_between_player) - sigmoid(1/pursuer_min_distance_to_obstacle)
            
            # check for special case where opponent is directly in front, yet behind an obstacle, so robot priotize obstacle avoidance
            elif state["Front"] == "Close" and state["Opponent Position"] == "Front" \
                and true_distance_between_player >= pursuer_min_distance_to_obstacle\
                and pursuer_min_distance_to_obstacle < true_safe_distance_from_obstacle\
                and pursuer_min_distance_to_obstacle_direction == "Front":

                state_description = "Evader is directly in front, but probably is behind an obstacle. Prioritize obstacle avoidance"
                reward = -0.25 - sigmoid(true_distance_between_player) - sigmoid(1/pursuer_min_distance_to_obstacle)
            
            # else if the evader is in front and closeby, and we are relatively safe from obstacles on either sides
            elif state["Opponent Position"] == "Front" and true_distance_between_player <= 1.0 and\
                true_distance_between_player <= pursuer_min_distance_to_obstacle:
                # encourage robot to orient itself such that the opponent is directly in front of it
                # take away the sigmoid of the distance to encourage it to minimize such distance 
                state_description = "Evader is in front and close enough by, and we are relatively safe from obstacle!"
                reward = sigmoid(1.0/true_distance_between_player)
            
            elif state["Opponent Position"] == "Front":
                state_description = "Evader is in front but not that close"
                reward = 0.5* sigmoid(1/true_distance_between_player)
            
            # if the other robot is nearby and there is an obstacle, there is a chance that obstacle 
            # may be the other robot, so we encourage those states
            # or if the distance between players are very close
            elif (((state["Upper Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left")  or \
                    (state["Upper Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
                    (state["Lower Left"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Left") or \
                    (state["Lower Right"] in ["Close", "Too Close"] and state["Opponent Position"] == "Close Right") or \
                    (state["Front"] in ["Close"] and state["Opponent Position"] == "Close Front")) and\
                        true_distance_between_player <= pursuer_min_distance_to_obstacle) or\
                            (true_distance_between_player <= true_safe_distance_from_obstacle):
                state_description = "Evader is nearby and we are relatively safe from obstacles"
                reward = 2.5 * sigmoid(1/true_distance_between_player)

            # there is no obstacle nearby and the target evader is far away
            elif state["Upper Left"] not in ["Close", "Too Close"] and state["Lower Left"] not in ["Close", "Too Close"]\
                and state["Upper Right"] not in ["Close", "Too Close"] and state["Lower Right"] not in ["Close", "Too Close"]\
                and state["Front"] != "Close" and true_distance_between_player >= safe_distance_from_obstacle:
                
                state_description = "Safe from obstacle but opponent is not nearby"
                reward = sigmoid(pursuer_min_distance_to_obstacle) - sigmoid(true_distance_between_player)  
            else:
                state_description = "Neutral state"
                reward = 0

        # REWARD FUNCTION FOR EVADER -----------------------------------------------------------------------------------------------------------------------------------
        elif self.agent_type == "evader":

            true_distance_between_player = (distance_between_players - pursuer_radius)
            true_safe_distance_from_obstacle = (evader_radius + safe_distance_from_obstacle)

            if evader_stuck or evader_was_stuck_but_rescued:
                state_description = "STUCK!"
                reward = -30
                is_terminal_state = True

            elif state["Opponent Position"] == "Tagged":
                state_description = "TAGGED!"
                reward = -30
                is_terminal_state = True 
            elif game_timeout:
                state_description = "Game Timeout"
                reward = (30 * sigmoid(true_distance_between_player))/(num_times_evader_stuck_in_episode + 1)

            # avoid obstacle on all sides
            elif (state["Front"] == "Close"):
                state_description = "Obstacle really close by in front! Prioritize obstacle avoidance"
                reward = -0.75 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif (state["Upper Left"] == "Too Close") or\
                (state["Upper Right"] == "Too Close") or\
                (state["Lower Left"] == "Too Close") or\
                (state["Lower Right"] == "Too Close"):
                state_description = "Obstacle really closeby on the side. Prioritize obstacle avoidance"
                # extra punishments depending on how close the pursuer is and how close the evader is to an obstacle
                reward = -0.5 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            # encourage keeping a safe distance from obstacle
            elif ((state["Upper Left"] == "OK" and state["Lower Left"] == "OK") or (state["Upper Right"] == "OK" and state["Lower Right"] == "OK"))\
                and true_distance_between_player >= 0.75:
                state_description = "Robot is maintaing safe distance from obstacle on either side"
                reward = 0.5 + sigmoid(evader_min_distance_to_obstacle) - sigmoid(1/true_distance_between_player)
            elif state["Opponent Position"] == "Front" and true_distance_between_player <= 1.0:
                # discourage evader from moving toward the pursuer when they are within 1.0 unit from each other
                state_description = "Pursuer is in front within 1.0 unit of distance! Go the opposite direction"
                reward = -0.25 - sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif state["Opponent Position"] == "Bottom" and true_distance_between_player >= 1.0:
                state_description = "Pursue is behind and far away"
                reward = 0.5 + sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif state["Opponent Position"] == "Bottom" and true_distance_between_player >= 0.75:
                state_description = "Pursue is behind and decent distance away"
                reward = 0.25 + sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif "Close" not in state["Opponent Position"] and true_distance_between_player >= 0.75:
                state_description = "Pursuer is not close by we are relatively safe from obstacle"
                reward = 0.5 + sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif "Close" not in state["Opponent Position"]:
                state_description = "Pursuer is in vincinity but we are relatively safe from obstacle"
                reward = sigmoid(true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            elif "Close" in state["Opponent Position"]:
                state_description = "Pursuer is extremely close. Run away!"
                reward = -1 - 2 *sigmoid(1.0/true_distance_between_player) - sigmoid(1.0/evader_min_distance_to_obstacle)
            else:
                state_description = "Neutral state"
                reward = 0.25

        if verbose:
            # rospy.loginfo("DISTANCE BTW PLAYER: {}, PURSUER_MIN_DIST_OBSTACLE = {}, TRUE_SAFE_DISTANCE_FROM_OBSTACLE = {}".format(TRUE_DISTANCE_BETWEEN_PLAYERS, PURSUER_MIN_DISTANCE_TO_OBSTACLE, TRUE_SAFE_DISTANCE_FROM_OBSTACLE))
            # rospy.loginfo("{}'s state is {}".format(self.agent_type, state))
            rospy.loginfo("{}'s reward state's is {}".format(self.agent_type, state_description))
            rospy.loginfo("{}'s reward is {}".format(self.agent_type, reward / 30))

        return (reward / 30.0, is_terminal_state)





        # # get in-game information
        # pursuer_radius = self.get_game_information("pursuer_radius")
        # evader_radius = self.get_game_information("evader_radius")
        # safe_distance_from_obstacle = self.get_game_information("safe_distance_from_obstacle")
        # pursuer_stuck = self.get_game_information("pursuer_stuck")
        # evader_stuck = self.get_game_information("evader_stuck")
        # pursuer_was_stuck_but_rescued = self.get_game_information("pursuer_was_stuck_but_rescued")
        # evader_was_stuck_but_rescued = self.get_game_information("evader_was_stuck_but_rescued")
        # pursuer_min_distance_to_obstacle = self.get_game_information("pursuer_min_distance_to_obstacle")
        # evader_min_distance_to_obstacle = self.get_game_information("evader_min_distance_to_obstacle")
        # game_timeout = self.get_game_information("game_timeout")
        # num_times_evader_stuck_in_episode = self.get_game_information("num_times_evader_stuck_in_episode")
        # distance_between_players = self.get_game_information("distance_between_players")

        # if self.agent_type == "pursuer":
        #     true_distance_between_player = (distance_between_players - evader_radius)
        #     true_safe_distance_from_obstacle = (pursuer_radius + safe_distance_from_obstacle)
        #     # terminal state
        #     if pursuer_stuck or pursuer_was_stuck_but_rescued:
        #         state_description = "STUCK!"
        #         reward = -1 
        #     elif true_distance_between_player <= 0.3:
        #         state_description = "TAGGED!"
        #         reward = 1 
        #     else:
        #         # if player's min_distance_to_obstacle < true_safe_distance_from_obstacle, (min_distance_to_obstace - true_safe_distance_from_obstacle) is negative
        #         # the more negative, the bigger the punishment to the overall reward. 
        #         # else if min_distance_to_obstacle >= true_safe_distance_from_obstacle, (min_distance_to_obstace - true_safe_distance_from_obstacle) is positive or zero
        #         # it will act as extra inceptive in the reward function!
        #         reward = 0.05 *(sigmoid(1.0/true_distance_between_player) + 0.1* sigmoid(pursuer_min_distance_to_obstacle - true_safe_distance_from_obstacle) - sigmoid(1.0/pursuer_min_distance_to_obstacle))
        # else:
        #     true_distance_between_player = (distance_between_players - pursuer_radius)
        #     true_safe_distance_from_obstacle = (evader_radius + safe_distance_from_obstacle)
        #     if evader_stuck or evader_was_stuck_but_rescued:
        #         state_description = "STUCK!"
        #         reward = -1 
        #     elif true_distance_between_player <= 0.3:
        #         # rospy.loginfo("TAGGED!")
        #         state_description = "TAGGED!"
        #         reward = -1 
        #     elif game_timeout:
        #         state_description = "GAME TIMEOUT"
        #         reward = 1/(num_times_evader_stuck_in_episode + 1)
        #     else:
        #         reward = 0.05 * (sigmoid(true_distance_between_player) + 0.1* sigmoid(evader_min_distance_to_obstacle - true_safe_distance_from_obstacle) - sigmoid(1.0/evader_min_distance_to_obstacle))
        
        # if verbose:
        #     rospy.loginfo("{} received reward: {}".format(self.agent_type, reward))

        # return reward
    
    def learn(self, epsilon, time_to_apply_action = 0.33):
        # learn using experience relay
        current_state_continuous = self.get_agent_state_function(self.agent_type, "continuous")
        # current_state_discrete = self.get_agent_state_function(self.agent_type, "discrete")
            
        rospy.loginfo("Epsilon: {}".format(epsilon))
        
        # get action A from S using policy
        chosen_action = self.get_policy(current_state_continuous, epsilon= epsilon, verbose = True)
        translation_speed, turn_action = chosen_action
        
        # take action A and move player, this would change the player's state
        self.agent_take_action_function(self.agent_type, translation_speed, turn_action, time_to_apply_action)
            
        # robot is now in new state S' 
        new_state_continuous = self.get_agent_state_function(self.agent_type, "continuous")
        new_state_discrete = self.get_agent_state_function(self.agent_type, "discrete")
        
        
        # robot now observes reward R(S') at this new state S'
        reward, is_terminal = self.reward_function(new_state_discrete)

        # learn with experience replay
        self._memorize_experience(current_state_continuous, chosen_action, new_state_continuous, reward, is_terminal)
        batch_results = self._experience_replay(mini_batch_size= self.batch_size)
        
        if batch_results == None:
            rmse_current_batch, avg_q_current_batch = None, None 
        else:
            rmse_current_batch, avg_q_current_batch = batch_results

        return reward, current_state_continuous, translation_speed, turn_action, rmse_current_batch, avg_q_current_batch
    
    
    def save_agent(self, agent_filename):
        with open("{}".format(agent_filename), "w") as file_pointer:
            file_pointer.seek(0) # evader_rescue_thread.join()
            file_pointer.write(pickle.dumps(self))

    @classmethod
    def load_agent(self, agent_filename):
        if (not os.path.isfile(agent_filename)):
            rospy.loginfo("{} file is not found in current present working directory".format(agent_filename))
            raise Exception("File not found!")
        else:
            with open(agent_filename, "rb") as file_pointer:
                agent = pickle.load(file_pointer)
                # rospy.loginfo("Loaded agent's first layer's weight: {}".format(agent.network_layers[0].weights))

            return agent
        

