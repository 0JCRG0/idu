from src.utils.env_helper import EnvHelper

env = EnvHelper.load_env_variables()
OAI_API_KEY = env.api_keys.openai
ANTROPIC_API_KEY = env.api_keys.anthropic
HF_SECRETS = env.api_keys.hf
