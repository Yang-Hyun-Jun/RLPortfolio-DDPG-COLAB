import DataManager
import Visualizer
import utils
import torch
import numpy as np
from Metrics import Metrics
from Environment import environment
from Agent import agent
from Network import Actor
from Network import Qnet

if __name__ == "__main__":
    stock_code = ["010140", "005930", "034220"]

    path_list = []
    for code in stock_code:
        path = utils.Base_DIR + "/" + code
        path_list.append(path)

    #test data load
    train_data, test_data = DataManager.get_data_tensor(path_list,
                                                        train_date_start="20090101",
                                                        train_date_end="20150101",
                                                        test_date_start="20170102",
                                                        test_date_end=None)

    #dimension
    state1_dim = 5
    state2_dim = 2
    K = 3

    #Test Model load
    actor = Actor(K=K)
    actor_target = Actor(K=K)
    critic = Qnet(K=K)
    critic_target = Qnet(K=K)

    balance = 15000000
    min_trading_price = 0
    max_trading_price = 500000

    #Agent
    environment = environment(chart_data=test_data)
    agent = agent(environment=environment,
                  actor=actor, actor_target=actor_target,
                  critic=critic, critic_target=critic_target, K=K,
                  lr=1e-3, tau=0.005, discount_factor=0.9, delta=0.005,
                  min_trading_price=min_trading_price,
                  max_trading_price=max_trading_price)

    #Model parameter load
    critic_path = utils.SAVE_DIR + "/Models" + "/DDPGPortfolio_critic.pth"
    actor_path = utils.SAVE_DIR + "/Models" + "/DDPGPortfolio_actor.pth"
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))

    #Test
    metrics = Metrics()
    agent.set_balance(balance)
    agent.reset()
    agent.environment.reset()
    agent.epsilon = 0
    state1 = agent.environment.observe()
    portfolio = agent.portfolio
    steps_done = 0

    while True:
        action, trading, confidence = agent.get_action(torch.tensor(state1).float().view(1, K, -1),
                                                       torch.tensor(portfolio).float().view(1, K+1, -1))

        _, next_state1, next_portfolio, reward, done = agent.step(trading, confidence)

        steps_done += 1
        state1 = next_state1
        portfolio = next_portfolio

        metrics.portfolio_values.append(agent.portfolio_value)
        metrics.profitlosses.append(agent.profitloss)

        if steps_done % 1 == 0:
            print(f"balance:{agent.balance}")
            print(f"stocks:{agent.num_stocks}")
            print(f"actions:{action}")
            print(f"portfolio:{agent.portfolio}")
        if done:
            print(f"model{agent.profitloss}")
            break

    #Benchmark: B&H
    agent.set_balance(15000000)
    agent.reset()
    agent.environment.reset()
    agent.delta = 0.0
    state1 = agent.environment.observe()
    portfolio = agent.portfolio
    while True:
        trading = np.ones(K)/K
        confidence = abs(trading)
        _, next_state1, next_portfolio, reward, done = agent.step(trading, confidence)

        state1 = next_state1
        portfolio = next_portfolio
        metrics.profitlosses_BH.append(agent.profitloss)
        if done:
            print(f"B&H{agent.profitloss}")
            break


    Vsave_path2 = utils.SAVE_DIR + "/" + "/Metrics" + "/Portfolio Value Curve_test"
    Vsave_path4 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss Curve_test"
    Msave_path1 = utils.SAVE_DIR + "/" + "/Metrics" + "/Portfolio Value_test"
    Msave_path2 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss_test"
    Msave_path3 = utils.SAVE_DIR + "/" + "/Metrics" + "/Profitloss B&H"
    Msave_path4 = utils.SAVE_DIR + "/" + "/Metrics" + "/Balances"

    metrics.get_portfolio_values(save_path=Msave_path1)
    metrics.get_profitlosses(save_path=Msave_path2)
    metrics.get_profitlosses_BH(save_path=Msave_path3)
    metrics.get_balances(save_path=Msave_path4)

    Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
    Visualizer.get_profitloss_curve(metrics.profitlosses,  metrics.profitlosses_BH, save_path=Vsave_path4)

