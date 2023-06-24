from gym.envs.registration import register
from gym.envs import registry

# delete if it's registered
env_name = 'Overcooked-v0'
if env_name in registry.env_specs:
    del registry.env_specs[env_name]

register(
    id=env_name,
    entry_point='overcooked_ai_py.mdp.overcooked_env:Overcooked',
)
