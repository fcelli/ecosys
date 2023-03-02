from gym.envs.registration import register

register(
    id='Ecosys-v0',
    entry_point='ecosys.environment:EcosysEnv',
    max_episode_steps=500
)
