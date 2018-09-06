from agent.agent import Agent
from functions import *
import sys
import time 
import json

# if len(sys.argv) != 4:
# 	print("Usage: python train.py [stock] [window] [episodes]")
# 	exit()
# 

stock_name, window_size, episode_count = "TRUE_1D_Train", 121, 50


# stock_name, window_size, episode_count = str(input("stock_name : ")), int(input("window_size : ")), int(input("episode_count : "))

model_name = ""
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 10
commission = 0.157/100 * 2 

mem_action = 0
sold_price = 0   
profit_result = {}
for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)
	total_profit = 0
	agent.inventory = []
	bought_price = 0 
	for t in range(l):
		action = agent.act(state)
		next_state = getState(data, t + 1, window_size + 1)
		reward = (data[t+1] - data[t]) *action
		total_profit += reward 

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)
		if t%2 == 0 : 
			print("action :  ", action)
			print("---------------------------------------------------------------")
			print("Reward : ", formatPrice(reward) ,"  Total Profit : ", formatPrice(total_profit))
			print("episode : ", e, "/", episode_count + 1, " process : ", t, "/", l) 
			mem_action = action 

	if e % 10 == 0:
		agent.model.save("agent/models/model"+ stock_name + "_" + str(e))
		profit_result["model"+ stock_name + "_" + str(e)] = total_profit 

with open("train_result.txt","w") as outfile:
	json.dump(profit_result, outfile)
