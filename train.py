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
batch_size = 5 # day to calculate reward 
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

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0
		print("action : ", action)
		if action == 1: # buy
			agent.inventory.append(data[t])
			reward = data[t] - data[t-1] - bought_price*commission
			print("episode : ", e)
			print("Buy: " + formatPrice(data[t]))
			print("Reward: " , reward)
			print("Total Profit : ", total_profit)
			print("--------------------------------")
			if mem_action == 0 :
				bought_price = data[t]
		elif action == 0 and len(agent.inventory) > 0: # sell	
			if mem_action == 1 : 			
				bought_price = agent.inventory.pop(0)
				sold_price = data[t]
				reward = max(data[t] - bought_price, 0)
				total_profit += data[t] - bought_price
				print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
				print("Reward: " , reward)
				print("Total Profit : ", total_profit)
				print("--------------------------------")
			elif sold_price != 0 : 
				reward = sold_price - data[t]
				print("Unhold " + formatPrice(data[t]) + " | Reward: " + formatPrice(reward))
			else : 
				diff = data[t] - data[t-1]
				if diff > data[t-1]*commission : 
					reward = -diff 
				elif diff < data[t-1]*commission:
					reward = abs(diff)
				else : 
					reward = abs(diff)
				print("Unhold : reward : ", reward)
		else : 
			diff = data[t] - data[t-1]
			if diff > data[t-1]*commission : 
				reward = -diff 
			elif diff < data[t-1]*commission:
				reward = abs(diff)
			else : 
				reward = abs(diff)
			print("Unhold : reward : ", reward)


		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)
		print("episode : ", e, "/", episode_count + 1, " process : ", t, "/", l) 
		mem_action = action 
	if e % 10 == 0:
		agent.model.save("agent/models/model"+ stock_name + "_" + str(e))
		profit_result["model"+ stock_name + "_" + str(e)] = total_profit 

with open("train_result.txt","w") as outfile:
	json.dump(profit_result, outfile)