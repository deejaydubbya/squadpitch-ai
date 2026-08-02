import os

# Unit/integration tests must be hermetic. This is set before test modules import
# Settings, so an unrelated developer .env cannot alter collection or results.
# Runtime keeps its strict default .env loading behavior.
os.environ["SQUADPITCH_AI_SETTINGS_ENV_FILE"] = "__DISABLED__"
