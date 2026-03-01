#Empirical Sensitivity of Collision Time in Truncated SHA-256 Under Controlled Input Distribution Bias

Overview:
This project implements a modular Monte Carlo framework to study collision-time behavior in truncated SHA-256 under controlled input distribution bias.
The simulation compares empirical collision times against the classical birthday bound approximation under:

Uniform input sampling
Reduced-support distributions
Bit-skewed (biased) distributions

The objective is to examine how entropy reduction influences observed collision-time scaling.

Repository Structure:
collision_sim/
|
|> main.py
|> config.py
|>  hash_engine.py
|>  input_models.py
|>  collision_detector.py
|>  experiment_runner.py
|>  analysis.py
|>  plotting.py

Requirements:
Python 3.9+

Install dependencies:
pip install -r requirements.txt

How To Run:
From the project directory:
python main.py

The program will:
Execute Monte Carlo collision simulations
Aggregate statistical results
Generate analysis figures
Save plots in the figures/ directory

Output:
Figures include:
Empirical vs theoretical collision scaling
Relative deviation analysis
Collision time vs effective entropy
Typical runtime: 5–15 seconds on a standard laptop.

Notes

This project analyzes probabilistic collision behavior in truncated regimes and does not claim practical weaknesses in SHA-256.
