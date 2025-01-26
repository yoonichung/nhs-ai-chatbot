import os
from pathlib import Path
from dotenv import load_dotenv

env_file = Path("../dev.env")
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
    print("Loaded environment variables from dev.env.")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
