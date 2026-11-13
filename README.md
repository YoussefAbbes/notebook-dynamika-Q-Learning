# MountainCar — Q-Learning & Reinforcement Learning

An Advanced Machine Learning project (ESPRIT, 4DAS): teach an agent to drive an under-powered car up a hill using only trial and error.

## The problem

A car sits in a valley and must reach the flag on the right hill. Its engine is too weak to climb directly, so it has to **swing back first to build momentum**. The reward is −1 per step, with no positive signal until the goal is reached.

| | |
|---|---|
| Environment | Gymnasium `MountainCar-v0` (and `MountainCarContinuous-v0` for TD3) |
| Observation | position, velocity |
| Actions | push left, do nothing, push right |
| Episode limit | 200 steps |

## Goals

| | Objective |
|---|---|
| **BO1 / DSO1** | Reproduce the reference paper's tabular Q-Learning results |
| **BO2 / DSO2** | Improve discretization, hyperparameters and exploration |
| **BO3 / DSO3** | Try Deep RL: DQN, PPO, and TD3 (continuous actions) |
| **BO4 / DSO4** | Compare performance, stability, training time and cost |

Success means an average reward ≥ −110 and a success rate ≥ 90 % at evaluation (ε = 0), reported as mean ± standard deviation over several seeds.

## Reference

*Modelling a Reinforcement Learning Agent for Mountain Car Problem Using Q-Learning With Tabular Discretization*, MysuruCon 2022 — [10.1109/mysurucon55714.2022.9972352](https://doi.org/10.1109/mysurucon55714.2022.9972352)

## Contents

- [`MountainCar_RL_QLearning.ipynb`](MountainCar_RL_QLearning.ipynb) — notebook (currently: Business Understanding)
- [`MountainCar_RL_Cadrage_Projet.md`](MountainCar_RL_Cadrage_Projet.md) — full project framing (in French): state of the art, BO, DSO, risks

## Method

Follows the Team Data Science Process (TDSP): Business Understanding → Data Understanding → Modeling → Deployment → Customer Acceptance.
