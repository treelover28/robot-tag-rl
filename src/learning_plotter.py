import matplotlib.pyplot as plt
import numpy as np


class Learning_Plotter:
    def __init__(self, train_type, total_episodes, episode_time_limit, training_algorithm_name, allow_player_manual_rescue = False):
        # create subplot of 2 rows, 3 columns

        if training_algorithm_name == "DQN":
            self.fig, self.ax = plt.subplots(3,3)
        else:
            self.fig, self.ax = plt.subplots(2,3)

        if train_type == "pursuer":
            title = "Pursuer Training Progress Dashboard"
        elif train_type == "evader":
            title = "Evader Training Progress Dashboard"
        else:
            title = "Both Evader and Pursuer Training Progress Dashboard"
        
        # figure's super title   
        self.fig.suptitle(title, fontsize=16)
        
        # plot learning curve as robot learns
        self.learning_curve, = self.ax[0,0].plot([],[], "r-", label=training_algorithm_name)
        self.test_curve, = self.ax[0,0].plot([],[], linestyle="-", marker="x", color="k", label="{} Test-Phase Reward".format(training_algorithm_name))
        self.ax[0,0].set_xlabel("Training episode")
        self.ax[0,0].set_ylabel("Accumulated rewards")
        self.ax[0,0].set_xlim(0 , total_episodes)
        self.ax[0,0].set_ylim(-100,100)
        self.ax[0,0].set_title("Accumulated Rewards vs Training episodes")
        self.ax[0,0].legend(loc="upper left")
        self.ax[0,0].axhline(y= 0, color = "g", linestyle = "-")
        # self.ax[0,0].autoscale(enable=True, axis = 'y')
        
        # subplot on row 0 column 1 shows details regarding how the robot is doing each test phase
        self.tag_curve, = self.ax[0,1].plot([],[], "g-", marker="x", label="Number of tags in test phase")
        self.stuck_curve, = self.ax[0,1].plot([],[], "r-", marker="x", label="Number of times stuck in test phase")
        self.timeout_curve, = self.ax[0,1].plot([],[], "b-",  marker="x", label="Number of timeouts in test phase")
        self.ax[0,1].set_xlabel("Number of Episodes")
        self.ax[0,1].set_ylabel("Number of scenarios in Test Phase")
        self.ax[0,1].set_xlim(0, total_episodes)
        self.ax[0,1].set_ylim(0, 40)
        self.ax[0,1].yaxis.set_ticks(np.arange(0, 41, 1))
        self.ax[0,1].set_title("Test Phase Details")
        self.ax[0,1].legend(loc="upper left")
       
        # go_left_when_opponent_bottom_curve, = ax[0,2].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
        # go_right_when_opponent_bottom_curve, = ax[0,2].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
        # go_front_when_opponent_bottom_curve, = ax[0,2].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
        # ax[0,2].set_xlabel("Number of Episodes")
        # ax[0,2].set_ylabel("Proportion of actions chosen")
        # ax[0,2].set_xlim(0, total_episodes)
        # ax[0,2].set_ylim(0, 1.0)
        # ax[0,2].legend(loc="upper left")
        # ax[0,2].set_title("Proportion of actions chosen when opponent is BEHIND")

       
        if train_type == "pursuer":
            self.average_distance_at_terminal_curve, = self.ax[0,2].plot([],[], "g-", marker="x", label = "Average distance at terminal state")
            self.ax[0,2].set_xlabel("Number of Episodes")
            self.ax[0,2].set_ylabel("Average distance between players")
            self.ax[0,2].set_xlim(0, total_episodes)
            self.ax[0,2].set_ylim(0, 6.0)
            self.ax[0,2].legend(loc="upper right")
            self.ax[0,2].set_title("Average distance between players after game ends")

        elif train_type == "evader" and not allow_player_manual_rescue:
            self.average_time_at_terminal_curve, = self.ax[0,2].plot([],[], "g-", marker="x", label = "Average time survived")
            self.ax[0,2].set_xlabel("Number of Episodes")
            self.ax[0,2].set_ylabel("Average time suvived")
            self.ax[0,2].set_xlim(0, total_episodes)
            self.ax[0,2].set_ylim(0, episode_time_limit)
            self.ax[0,2].legend(loc="upper right")
            self.ax[0,2].set_title("Average time survived by evader")
        else:
            self.num_evader_stuck_curve, = self.ax[0,2].plot([],[], "g-", marker="x", label = "Average number of time evader got stuck")
            self.ax[0,2].set_xlabel("Number of Episodes")
            self.ax[0,2].set_ylabel("Average number of time stuck")
            self.ax[0,2].set_ylim(0, 10)
            self.ax[0,2].set_xlim(0, total_episodes)
            self.ax[0,2].legend(loc="upper right")
            self.ax[0,2].set_title("Average numer of times evader got stuck")


        self.go_left_when_opponent_left_curve, = self.ax[1,0].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
        self.go_right_when_opponent_left_curve, = self.ax[1,0].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
        self.go_front_when_opponent_left_curve, = self.ax[1,0].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
        self.ax[1,0].set_xlabel("Number of Episodes")
        self.ax[1,0].set_ylabel("Proportion of actions chosen")
        self.ax[1,0].set_xlim(0, total_episodes)
        self.ax[1,0].set_ylim(0, 1.0)
        self.ax[1,0].legend(loc="upper left")
        self.ax[1,0].set_title("Proportion of actions chosen when opponent is to the LEFT")

        self.go_left_when_opponent_right_curve, = self.ax[1,1].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
        self.go_right_when_opponent_right_curve, = self.ax[1,1].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
        self.go_front_when_opponent_right_curve, = self.ax[1,1].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
        self.ax[1,1].set_xlabel("Number of Episodes")
        self.ax[1,1].set_ylabel("Proportion of actions chosen")
        self.ax[1,1].set_xlim(0, total_episodes)
        self.ax[1,1].set_ylim(0, 1.0)
        self.ax[1,1].legend(loc="upper left")
        self.ax[1,1].set_title("Proportion of actions chosen when opponent is to the RIGHT")


        self.go_left_when_opponent_front_curve, = self.ax[1,2].plot([], [], "g-", marker="x", label = "Left Turn Proportion")
        self.go_right_when_opponent_front_curve, = self.ax[1,2].plot([], [], "r-", marker="x", label = "Right Turn Proportion")
        self.go_front_when_opponent_front_curve, = self.ax[1,2].plot([], [], "b-", marker="x", label = "Go Straight Proportion")
        self.ax[1,2].set_xlabel("Number of Episodes")
        self.ax[1,2].set_ylabel("Proportion of actions chosen")
        self.ax[1,2].set_xlim(0, total_episodes)
        self.ax[1,2].set_ylim(0, 1.0)
        self.ax[1,2].legend(loc="upper left")
        self.ax[1,2].set_title("Proportion of actions chosen when opponent is IN FRONT")

        if training_algorithm_name == "DQN":
            # plot average Q-values to see Q-convergence
            self.avg_q_curve, = self.ax[2,0].plot([],[], "g-", marker="o", label="Average Q-value across all actions")
            self.ax[2,0].set_xlabel("Number of episodes")
            self.ax[2,0].set_ylabel("Average Q-value")
            self.ax[2,0].set_xlim(0, total_episodes)
            self.ax[2,0].set_ylim(-10,10)
            self.ax[2,0].legend(loc="upper left")
            self.ax[2,0].set_title("Average Q-value of all actions across training episodes")

        
            self.avg_loss_curve,  = self.ax[2,1].plot([],[], "g-", marker="x", label = "Root Mean Squared Error/Loss Per Episode")
            self.ax[2,1].set_xlabel("Number of Episodes")
            self.ax[2,1].set_ylabel("Root Mean Squared Loss")
            self.ax[2,1].set_xlim(0, total_episodes)
            self.ax[2,1].set_ylim(0, 10)
            self.ax[2,1].legend(loc="upper right")
            self.ax[2,1].set_title("RMSE Per Episode of Q-Network's predictions")

        plt.show(block=False)


    def plot_learning_curve(self, line_chart, new_x, new_y):
        
        line_chart.set_xdata(np.append(line_chart.get_xdata(), new_x))
        line_chart.set_ydata(np.append(line_chart.get_ydata(), new_y))
        plt.draw()
        plt.pause(0.001)


    def savefig(self, figure_name, dpi):
        plt.savefig(figure_name, dpi=dpi)