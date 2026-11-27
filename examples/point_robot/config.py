from legged_gym.envs.base.base_config import BaseConfig
class Config(BaseConfig):
    seed=42
    class env:
        num_observations=4
        num_actions=2
    class Train(BaseConfig):
        seed=42
        policy=dict(actor_hidden_dims=[32,32],critic_hidden_dims=[32,32],activation='elu',init_noise_std=0.1)
        algorithm=dict()
        class runner:
            policy_class_name='ActorCritic'
            algorithm_class_name='PPO'
            max_iterations=10
            num_steps_per_env=32
            save_interval=5
            run_name=''
            experiment_name='point_robot'
