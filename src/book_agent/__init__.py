"""book-agent 패키지."""


def main() -> None:
    # 무거운 import(langchain, chroma, torch)는 실제로 실행할 때만 일어나도록 지연한다.
    from .app import main as run_app

    run_app()
