# tf2rl
Deep Reinforcement Learning in TF2.0


# Algorithms
### Model Free
- [x] [DDPG](tf2rl/ddpg/ddpg.py)
- [x] [TD3](tf2rl/tf3/td3.py)

#### How to train ?
Here's an example to train TD3 algorithm on the LunarLander environment

```
from tf2rl.td3.policies import TD3Policy
from tf2rl.td3.td3 import TD3

env = gym.make("LunarLanderContinuous-v2")
model = TD3(TD3Policy, env, log_dir_path="td3_results_3")
model.learn(total_timesteps=1000000)

```

# Credits
This code was inspired by the [stable-baselines](https://github.com/hill-a/stable-baselines) repo.
The main contribution of this work is the support of Tensorflow 2.
