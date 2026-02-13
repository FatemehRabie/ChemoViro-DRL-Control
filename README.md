# Optimal Control of Virotherapy-Chemotherapy via Deep Reinforcement Learning

This repository contains the code and experiments for the paper **"Optimal Control of a Virotherapy-Chemotherapy Combination Model via Deep Reinforcement Learning: A Comprehensive Framework"**.

## 🧬 Overview
This project presents a comprehensive framework for applying Deep Reinforcement Learning (DRL) to solve optimal control problems in cancer therapy models (specifically combined Onco-Virotherapy and Chemotherapy). By formulating the continuous-time ODE system as a Markov Decision Process (MDP), we train agents to autonomously discover effective, multi-stage treatment protocols.

## 🚀 Key Features
* **Custom Gymnasium Environment:** A simulation environment based on nonlinear ODEs (Reduced 4D and Full 6D models) with configurable parameters.
* **State-of-the-Art Algorithms:** Implementation of PPO, TRPO, SAC, and TD3 using `Stable-Baselines3`.
* **Robustness Analysis:** Extensive testing across different scenarios, including:
    * **Standard Conditions:** Benchmarking against classical Dynamic Programming.
    * **Aggressive Tumor Dynamics:** A stress-test scenario where RL discovers non-intuitive optimal strategies (e.g., non-intervention when spontaneous regression occurs) that classical fixed policies fail to identify.
    * **Model Mismatch & Noise:** Evaluating policy generalization from simplified to complex biological models.
* **Hyperparameter Tuning:** Automated tuning pipeline using `Optuna`.

## 📊 Results Highlights
Our experiments demonstrate that DRL agents, particularly on-policy methods like TRPO and PPO, achieve superior generalization and robustness compared to classical baselines. The agents successfully learn adaptive closed-loop policies that can handle environmental uncertainty and model mismatch.

## 🛠️ Installation & Usage
(Add your installation instructions here)
