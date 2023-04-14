import gym
import numpy as np
import pandas as pd
from scipy.integrate import odeint

def sir(y, t, beta, gamma, u):
    S, I = y
    dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
    return dydt

class SirEnvironment(gym.Env):
    population: float
    beta: float
    gamma: float
    v_min: float
    v_max: float
    S0: float
    I0: float
    tf: float
    dt: float
    days: list
    susceptible: list
    infected: list
    actions: list
    vaccine_importance: float
    continuous: bool
    def __init__(self, S0, I0, beta, gamma, v_min, v_max, tf, dt, vaccine_importance, continuous, population):
        self.state = np.array([S0, I0])
        self.beta = beta
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.S0 = S0
        self.I0 = I0
        self.tf = tf
        self.vaccine_importance = vaccine_importance
        self.continuous = continuous
        self.observation_space = gym.spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([population, population], dtype=np.float32),
                    dtype=np.float32)

        if self.continuous:
            self.action_space = gym.spaces.Box(
                        low=np.array([-1.0], dtype=np.float32),
                        high=np.array([1.0], dtype=np.float32),
                        dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(2)

        self.time = 0.0
        self.dt = dt

    def reset(self):
        self.time = 0.0
        self.days = [self.time]
        self.susceptible = [self.S0]
        self.infected = [self.I0]
        self.actions = []
        self.rewards = []
        self.state = np.array([self.S0, self.I0])
        return np.array(self.state, dtype=np.float32)

    def action2vaccine(self, action):
        return self.v_min + self.v_max * (action[0] + 1.0) / 2.0

    def step(self, action):
        if self.continuous:
            vaccine = self.action2vaccine(action)
        else:
            vaccine = self.v_min if action == 0 else self.v_max
        self.actions.append(vaccine)

        sol = odeint(sir, self.state, np.linspace(0, self.dt, 101), args=(self.beta, self.gamma, vaccine))
        self.time += self.dt
        new_state = sol[-1, :]
        S0, I0 = self.state
        S, I = new_state
        self.state = new_state
        reward = - I - self.vaccine_importance * vaccine
        reward *= self.dt

        self.rewards.append(reward)
        self.days.append(self.time)
        self.susceptible.append(S)
        self.infected.append(I)

        done = True if self.time >= self.tf else False
        return (np.array(new_state, dtype=np.float32), reward, done, {})

    @property
    def dynamics(self):
        df= pd.DataFrame(dict(
                                days=self.days,
                                susceptible=self.susceptible,
                                infected=self.infected,
                                vaccines=self.actions + [None],
                                rewards=self.rewards + [None]
                            )
                        )
        return df

def sliar(y, t, beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, nu):
    S, L, I , A = y
    dydt = np.array([- beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - nu * S,
                    beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - kappa * L,
                    p * kappa * L - alpha * I - tau * I,
                    (1 - p) * kappa * L  - eta * A])
    return dydt

class SliarEnvironment(gym.Env):
    population: float
    beta: float
    sigma_min: float
    sigma_max: float
    kappa_min: float
    kappa_max: float
    tau_min: float
    tau_max: float
    tau: float
    p: float
    eta: float
    epsilon: float
    q: float
    delta: float
    S0: float
    I0: float
    L0: float
    A0: float
    tf: float
    dt: float
    P: float
    Q: float
    R: float
    W: float
    def __init__(self, S0, I0, L0, A0, R0,
                 sigma_min, sigma_max, nu_min, nu_max, kappa,
                 alpha, tau_min, tau_max, p, eta, epsilon,
                 q, delta, tf, dt, population, P, Q, R, W, continuous):
        self.state = np.array([S0, L0, I0, A0])
        self.population = population
        self.S0 = S0
        self.I0 = I0
        self.L0 = L0
        self.A0 = A0
        self.beta = R0 / (S0 * ((epsilon / kappa) + ((1 - q)*p/alpha) + (delta*(1-p)/eta)))
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kappa = kappa
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.alpha = alpha
        self.p = p
        self.eta = eta
        self.epsilon = epsilon
        self.q = q
        self.delta = delta
        self.tf = tf
        self.dt = dt
        self.P = P
        self.Q = Q
        self.R = R
        self.W = W
        self.continuous = continuous

        self.observation_space = gym.spaces.Box(
                    low=np.array([0.0]*4, dtype=np.float32),
                    high=np.array([population]*4, dtype=np.float32),
                    dtype=np.float32)
        if self.continuous:
            self.action_space = gym.spaces.Box(
                            low=np.array([-1.0]*3, dtype=np.float32),
                            high=np.array([1.0]*3, dtype=np.float32),
                            dtype=np.float32)
        else:
            self.action_space = gym.spaces.MultiBinary(3)

        self.time = 0.0
        self.dt = dt

    def reset(self):
        self.time = 0.0
        self.days = [self.time]
        self.susceptible = [self.S0]
        self.latent = [self.L0]
        self.infected = [self.I0]
        self.asymp = [self.A0]
        self.nus = []
        self.taus = []
        self.sigmas = []
        self.rewards = []
        self.state = np.array([self.S0, self.L0, self.I0, self.A0])
        return np.array(self.state, dtype=np.float32), {}

    def action2control(self, action):
        if self.continuous:
            nu = self.nu_min + (self.nu_max - self.nu_min) * (action[0] + 1.0) / 2.0
            tau = self.tau_min + (self.tau_max - self.tau_min) * (action[1] + 1.0) / 2.0
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (action[2] + 1.0) / 2.0
        else:
            nu = self.nu_min + (self.nu_max - self.nu_min) * action[0]
            tau = self.tau_min + (self.tau_max - self.tau_min) * action[1]
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * action[2]
        return np.array([nu, tau, sigma], dtype=np.float32)

    def step(self, action):
        nu, tau, sigma = self.action2control(action)
        self.nus.append(nu)
        self.taus.append(tau)
        self.sigmas.append(sigma)

        sol = odeint(sliar, self.state, np.linspace(0, self.dt, 101),
                     args=(self.beta, sigma, self.kappa, self.alpha, tau, self.p, self.eta, self.epsilon, self.q, self.delta, nu))

        self.time += self.dt
        new_state = sol[-1, :]
        S0, L0, I0, A0 = self.state
        S, L, I, A = new_state
        self.state = new_state
        reward = - self.P * (I / self.population) - self.Q * (nu / self.nu_max) ** 2 \
                 - self.R * (tau / self.tau_max) ** 2 - self.W * (sigma / self.sigma_max) ** 2
        reward *= self.dt

        self.rewards.append(reward)
        self.days.append(self.time)
        self.susceptible.append(S)
        self.latent.append(L)
        self.infected.append(I)
        self.asymp.append(A)

        done = True if self.time >= self.tf else False
        return (np.array(new_state, dtype=np.float32), reward, done, {})

    @property
    def dynamics(self):
        df = pd.DataFrame(dict(
                                days=self.days,
                                susceptible=self.susceptible,
                                latent=self.latent,
                                infected=self.infected,
                                asymp=self.asymp,
                                nus=self.nus + [None],
                                taus=self.taus + [None],
                                sigmas=self.sigmas + [None],
                                rewards=self.rewards + [None])

                        )
        return df

def seiar(y, t, beta, psi, nu, kappa, alpha, tau, p, eta, f, epsilon, q, delta):
    S, E, I, A, R = y
    Lambda = epsilon * E + (1 - q) * I + delta * A
    dSdt = -beta * S * Lambda - psi * nu * S
    dEdt = beta * S * Lambda - kappa * E
    dIdt = p * kappa * E - alpha * I - tau * I
    dAdt = (1 - p) * kappa * E - eta * A
    dRdt = f * alpha * I + tau * I + eta * A + psi * nu * S
    return [dSdt, dEdt, dIdt, dAdt, dRdt]

class SeiarEnvironment(gym.Env):
    beta: float
    psi: float
    nu_daily_max: float
    nu_total_max: float
    kappa: float
    alpha: float
    tau: float
    p: float
    eta: float
    f: float
    epsilon: float
    q: float
    delta: float
    S0: float
    E0: float
    I0: float
    A0: float
    R0: float
    tf: float

    def __init__(self, beta, psi, nu_daily_max, nu_total_max, kappa, alpha,
                 tau, p, eta, f, epsilon, q, delta,
                 S0, E0, I0, A0, R0, tf, dt):
        super(SeiarEnvironment, self).__init__()
        self.beta = beta
        self.psi = psi
        self.nu_daily_max = nu_daily_max
        self.nu_total_max = nu_total_max
        self.nu_min = 0.0
        self.kappa = kappa
        self.alpha = alpha
        self.tau = tau
        self.p = p
        self.eta = eta
        self.f = f
        self.epsilon = epsilon
        self.q = q
        self.delta = delta
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.A0 = A0
        self.R0 = R0
        self.tf = tf
        self.dt = dt
        self.time = 0
        self.days = [self.time]
        self.history = [[S0, E0, I0, A0, R0]]
        self.nus = []
        self.rewards = []
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.time = 0
        self.days = [self.time]
        self.state = np.array([self.S0, self.E0, self.I0, self.A0, self.R0])
        self.nus = []
        self.rewards = []
        self.history = [self.state]
        return np.array(self.state, dtype=np.float32)

    def action2control(self, action):
        nu = self.nu_min + (self.nu_daily_max - self.nu_min) * (action[0] + 1.0) / 2.0
        return nu

    def step(self, action):        
        nu = self.action2control(action)
        self.nus.append(nu)
        S0, E0, I0, A0, R0 = self.state
        sol = odeint(seiar, [min(0, S0-nu), E0, I0, A0, R0], 
                     np.linspace(0, self.dt, 101),
                     args=(self.beta, self.psi, 0,
                           self.kappa, self.alpha, self.tau, 
                           self.p, self.eta, self.f, self.epsilon,
                           self.q, self.delta))

        self.time += self.dt
        new_state = sol[-1, :]
        S, E, I, A, R = new_state
        self.state = new_state

        reward = - I - nu
        if np.sum(self.nus) > self.nu_total_max:
            reward -= 1000
        reward *= self.dt

        self.rewards.append(reward)
        self.days.append(self.time)
        self.history.append([S, E, I, A, R])

        done = True if self.time >= self.tf else False
        return (np.array(new_state, dtype=np.float32), reward, done, {})
    
    @property
    def dynamics(self):
        df = pd.DataFrame(dict(
                                days=self.days,
                                susceptible=[s[0] for s in self.history],
                                infected=[s[2] for s in self.history],
                                nus=self.nus + [None],
                                rewards=self.rewards + [None])

                        )
        return df
