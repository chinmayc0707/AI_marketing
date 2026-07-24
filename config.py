import os
from dotenv import load_dotenv, set_key

DOTENV_PATH = os.path.join(os.path.dirname(__file__), '.env')

APP_TITLE = "Coca-Cola Marketing Generator"
APP_REFERER = "http://localhost:5000"

SYSTEM_PROMPT = (
    "You are an elite, high-converting growth marketer and copywriter.\n"
    "IMPORTANT FORMATTING INSTRUCTIONS:\n"
    "1. Put every final copy inside a markdown code block (using ```) so the user can easily copy individual quotes with one click.\n"
    "2. Separate distinct options or sections with horizontal line dividers (using ---)."
)

def load_environment():
    """Ensure environment variables are loaded and up to date."""
    load_dotenv(DOTENV_PATH, override=True)

def get_env_model() -> str:
    load_environment()
    return os.getenv("LLM_MODEL", "").strip("'\"").strip()

def get_api_key() -> str:
    load_environment()
    return os.getenv("OPENROUTER_API_KEY", "").strip("'\"").strip()

def update_api_key(api_key: str):
    if api_key and api_key.strip():
        clean_key = api_key.strip().strip("'\"")
        os.environ['OPENROUTER_API_KEY'] = clean_key
        set_key(DOTENV_PATH, 'OPENROUTER_API_KEY', clean_key)

def get_masked_api_key() -> tuple[bool, str]:
    key = get_api_key()
    has_key = bool(key)
    if not has_key:
        return False, ""
    if len(key) > 8:
        masked = key[:4] + "..." + key[-4:]
    else:
        masked = "********"
    return True, masked
