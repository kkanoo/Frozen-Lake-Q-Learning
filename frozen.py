import gym  #importing the gym library
import numpy as np
import time 
import random
from IPython.display import clear_output #to clear output

class Frozen(): #creating the class Frozen
    
    def __init__(self): #constructor
        self.env = gym.make('FrozenLake-v0') #FrozenLake Environment from gym module
        self.action_space_size = self.env.action_space.n #storing the number of actions that can be taken in frozen lake environment
        self.state_space_size = self.env.observation_space.n #storing the number of states in the environment
        self.q_table = np.zeros((self.state_space_size, self.action_space_size)) #creating a q value table with rows as total no. of state and column as number of actions that can be taken

        self.number_episodes = 10000 #Epochs
        self.max_steps_per_episode = 100 #if the goal is not reached in 100 steps in an episode that epoch will terminate and new will start

        self.learning_rate = 0.1  # alpha value in the equation of temporal difference
        self.discount_factor = 0.99 # gamma value in the bellman equation

        self.exploration_rate = 1  # epsilon greedy (1 means it will only explore and not exploite)
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001

        self.rewards_all_episodes = []  #will contain all the rewards gained in 10000 episodes
        
    def q_values(self):  #calculating the q_values in 10000 episodes

        for episode in range(self.number_episodes):  #loopng through 10000 episodes (epochs)
            state = self.env.reset()   #stroing the current state of the environment
            
            done = False     #whether the goal is reached or not. (set false initially)
            rewards_current_episode = 0   #reward of one episode (use to append rewards_all_episodes)
            
            for step in range(self.max_steps_per_episode):  #dividing the 10000 episodes to 100 episodes at a time
                
                exploration_rate_threshold = random.uniform(0,1) #setting a threshold for exploration or exploitation
                if exploration_rate_threshold > self.exploration_rate: #if threshold is greater than rate, exploitation will be done, i.e., action will be taken according to the max q value in that state
                    action = np.argmax(self.q_table[state, :]) #max Q value for that state
                else:   
                    action = self.env.action_space.sample()  # any action is choosen for that state i.e., exploration
                    
                new_state, reward, done, info = self.env.step(action) #returns a tuple containing the new state, reward, wheather the agent fell in the hole or finished the task i.e., reached the goal
                
                self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + self.learning_rate*(reward + self.discount_factor*np.max(self.q_table[new_state, :])) #using temporal difference to calculate new Q value for that state for that particular action
                
                state = new_state  #updating the state
                rewards_current_episode += reward  #updating the reward for 100 episodes everytime
                
                if done == True:  #if reached the goal or fell through the hole
                    break
                
            #calculating the exploration rate decay after each episode
            self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-self.exploration_decay_rate*episode)
            self.rewards_all_episodes.append(rewards_current_episode)
        #calculate and print average reward per thousand episodes
        reward_per_thousand_episode = np.split(np.array(self.rewards_all_episodes), self.number_episodes/1000)
        count = 1000
        for r in reward_per_thousand_episode:
            print(count, ": ", str(sum(r/1000))) #printing the reward gained per 1000 episodes
            count += 1000
            
        print("*************Q-Table*****************")    
        print (self.q_table)  #printing the q_values table
        
    def show(self):  #visualising the game played by the AI

        for episode in range(3):  #3 round game
            state = self.env.reset()  #initial state of the environment
            done = False
            print("*******EPISODE", episode+1,"******\n\n\n\n")
            time.sleep(1) #wait for one second to see the output
            
            for step in range(self.max_steps_per_episode):  #100 episodes
                clear_output(wait=True)
                self.env.render() #to show the current state
                time.sleep(0.3)
                
                action = np.argmax(self.q_table[state, :])  #choosing the action w.r.t to explotation
                new_state, reward, done, info = self.env.step(action)  #getting the new_state, reward, done and info
                
                if done:  #if goal is reached or fell through the hole
                    clear_output(wait=True)
                    self.env.render()  #to visually show the current state
                    if reward == 1:  #if the current state reward is 1 then the goal is reached
                        print("You've reached your goal")
                        time.sleep(3)
                    else: #if not it fell through a hole
                        print("You fell through a hole")
                        time.sleep(3)
                    clear_output(wait=True)
                    break
                
                state = new_state #updating the state
                 
        self.env.close()  #closing the environment
                
frozen = Frozen()  #object of the Frozen class
frozen.q_values()  #calling q_values function
frozen.show()      #calling show function