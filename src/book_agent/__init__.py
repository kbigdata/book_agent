'''book-agent 패키지.'''

def main() -> None:
    from .device import resolve_device
    from .settings import load_settings

    settings=load_settings()

    print(f"model={settings.llm_model}, device={resolve_device(settings.device)}")
