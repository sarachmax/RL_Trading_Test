from agent.agent import Agent
from functions import *
import sys
import time 

# if len(sys.argv) != 4:
# 	print("Usage: python train.py [stock] [window] [episodes]")
# 	exit()
# 
stock_name, window_size, episode_count = "^GSPC", 121, 50


# stock_name, window_size, episode_count = str(input("stock_name : ")), int(input("window_size : ")), int(input("episode_count : "))


agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

mem_action = 0
sold_price = 0   
for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)
	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0
		print("action : ", action)
		if action == 1: # buy
			agent.inventory.append(data[t])
			print("episode : ", e)
			print("Buy: " + formatPrice(data[t]))
			print("--------------------------------")
		elif action == 0 and len(agent.inventory) > 0: # sell	
			if mem_action == 1 : 			
				bought_price = agent.inventory.pop(0)
				sold_price = data[t]
				reward = max(data[t] - bought_price, 0)
				total_profit += data[t] - bought_price
				print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
				print("--------------------------------")
			elif sold_price != 0 : 
				reward = sold_price - data[t]
				print("Unhold " + formatPrice(data[t]) + " | Reward: " + formatPrice(reward))
		else : 
			print("Action : Unhold") 
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

	if e % 10 == 0:
		agent.model.save("models/model_ep"+ stock_name + "_" + str(e))
