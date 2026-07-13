"""Persisted personal settings -- currently just how you want things explained.
Stored on K:\\AI-Mentor so it survives restarts and applies across all modes."""
import json

from config import DATA_DIR, ensure_dirs

SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULT_EXPLANATION_STYLE = (
    "Explain like you're prepping me for an AI engineer interview: give a clear, "
    "concrete analogy before the formal definition, connect the idea to how it "
    "would actually come up in an interview or on the job, and avoid jargon "
    "unless you define it immediately. Prefer plain language over dense theory."
)


def get_settings() -> dict:
    ensure_dirs()
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return {"explanation_style": DEFAULT_EXPLANATION_STYLE}


def update_settings(explanation_style: str | None = None) -> dict:
    settings = get_settings()
    if explanation_style is not None:
        settings["explanation_style"] = explanation_style
    ensure_dirs()
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings
