import numpy as np

seed = 2
#넘파이 랜덤 시드 고정
np.random.seed(seed)


class OUProcess:
  def __init__(self, mu):
      self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
      self.mu = mu
      self.x_prev = np.zeros_like(self.mu)

  def __call__(self):
      x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
          self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
      self.x_prev = x
      return x


class Normal:
    def __init__(self, mu, std, size):
        self.mu = mu
        self.std = std
        self.size = size

    def __call__(self):
        x = np.random.normal(loc=self.mu, scale=self.std, size=self.size)
        return x
