from gymnasium.envs.registration import register

register(
    id="affine_gym_env/AffineEnv",
    entry_point="affine_gym_env.envs.affine_env:AffineEnv",
    max_episode_steps=500,
)