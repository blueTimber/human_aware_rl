from gym.envs.registration import register

register(
    id='Overcooked-v1',
    entry_point='overcooked_ai_py.mdp.overcooked_env:Overcooked',
)