# Component 4: Safe Deep Reinforcement Learning for Autonomous Storage Optimization

## Research Scope
Executes autonomous storage optimization actions (load balancing, preventive spin-down, data migration) under safety constraints using Safe Reinforcement Learning.

## Structure
- `data/`: Independent simulation logs, states, and transition splits.
- `models/`: Safe RL policies and constrained action evaluators.
- `src/`: Environment wrapper and policy enforcement logic.
- `config/`: Configuration parameters (`paths.yaml`, `component_config.yaml`).
