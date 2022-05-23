import Visualizer
import utils
import torch
import numpy as np
from Metrics import Metrics
from Environment import environment
from Agent import agent
from Network import Actor
from Network import Qnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGTester:
    def __init__(self, test_data, balance, min_trading_price, max_trading_price, delta, K):

        self.test_data = test_data

        self.state1_dim = 5
        self.state2_dim = 2
        self.K = K

        self.actor = Actor(K=K).to(device)
        self.actor_target = Actor(K=K).to(device)
        self.critic = Qnet(K=K).to(device)
        self.critic_target = Qnet(K=K).to(device)

        self.delta = delta
        self.balance = balance
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        #Agent
        self.env = environment(chart_data=test_data)
        self.agent = agent(environment=self.env,
                           actor=self.actor, actor_target=self.actor_target,
                           critic=self.critic, critic_target=self.critic_target, K=self.K,
                           lr=0.0, tau=0.0, discount_factor=0.0, delta=self.delta,
                           min_trading_price=self.min_trading_price,
                           max_trading_price=self.max_trading_price)

        #Model parameter load
        critic_path = utils.SAVE_DIR + "/Models" + "/DDPGPortfolio_critic.pth"
        actor_path = utils.SAVE_DIR + "/Models" + "/DDPGPortfolio_actor.pth"
        self.agent.actor.load_state_dict(torch.load(actor_path))
        self.agent.critic.load_state_dict(torch.load(critic_path))


    def run(self):
        metrics = Metrics()
        self.agent.set_balance(self.balance)
        self.agent.reset()
        self.agent.environment.reset()

        state1 = self.agent.environment.observe()
        portfolio = self.agent.portfolio
        steps_done = 0

        while True:
            action, trading, confidence = self.agent.get_action(torch.tensor(state1, device=device).float().view(1, self.K, -1),
                                                                torch.tensor(portfolio, device=device).float().view(1, self.K+1, -1))

            _, next_state1, next_portfolio, reward, done = self.agent.step(trading, confidence)

            steps_done += 1
            state1 = next_state1
            portfolio = next_portfolio

            metrics.portfolio_values.append(self.agent.portfolio_value)
            metrics.profitlosses.append(self.agent.profitloss)

            if steps_done % 1 == 0:
                print(f"balance:{self.agent.balance}")
                print(f"stocks:{self.agent.num_stocks}")
                print(f"actions:{action}")
                print(f"portfolio:{self.agent.portfolio}")
            if done:
                print(f"model{self.agent.profitloss}")
                break


        #Benchmark: B&H
        self.agent.set_balance(self.balance)
        self.agent.reset()
        self.agent.environment.reset()
        self.agent.delta = 0.0
        self.agent.environment.observe()
        while True:
            trading = np.ones(self.K)/self.K
            confidence = abs(trading)

            _, next_state1, next_portfolio, reward, done = self.agent.step(trading, confidence)
            metrics.profitlosses_BH.append(self.agent.profitloss)

            if done:
                print(f"B&H{self.agent.profitloss}")
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
