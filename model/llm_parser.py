import ollama, json, time, hashlib
from pathlib import Path

MODEL = "llama3.2:latest"
CACHE_DIR = Path(".llm_cache")
TEMPLATE_FILE = Path("prompt_template.txt")
REQUIRED = {"day", "session_start", "session_end", "fuzzy_type"}

CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def _load_template() -> str:
    return TEMPLATE_FILE.read_text()

def _call_ollama(messages, retries=5, wait=2):
    for i in range(retries):
        try:
            resp = ollama.chat(
                model    = MODEL,
                messages = messages,
                options  = {
                    "temperature": 0,
                    "response_format": { "type": "json_object" }
                },
                stream   = False
            )
            txt = resp["message"]["content"].strip()
            if txt:
                return txt
        except Exception as e:
            print(f"[LLM] attempt {i+1}/{retries} failed – {e}")
        time.sleep(wait)
    raise RuntimeError("Ollama never returned usable content")

def parse_with_llm(phrase: str, course_key: str, duration: int, refresh=False):
    cache_key = f"{course_key}|{duration}|{phrase}"
    p = _cache_path(cache_key)
    if not refresh and p.exists():
        return json.load(p.open())

    system_msg = _load_template().format(course_duration=duration)
    user_msg   = f'Course {course_key} (duration {duration}) preference: "{phrase}"'

    raw = _call_ollama([
        { "role": "system", "content": system_msg },
        { "role": "user",   "content": user_msg }
    ])

    data = json.loads(raw)
    if not REQUIRED.issubset(data):
        raise ValueError(f"LLM output missing keys: {data}")

    json.dump(data, p.open("w"))
    return data