"""
목차(outline) 저장소 모듈.

노드는 OutlineRepository 프로토콜에만 의존하므로
파일 / DB / 메모리 등 저장 방식을 바꿔도 노드 코드는 그대로다.
"""

from pathlib import Path
from typing import Protocol

EMPTY_OUTLINE = '아직 작성된 목차가 없습니다.'


class OutlineRepository(Protocol):
    def load(self) -> str: ...

    def save(self, outline: str) -> None: ...


class FileOutlineRepository:
    """목차를 마크다운 파일 하나로 저장한다. 사용자는 이 파일을 열어 목차를 확인한다."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> str:
        if not self._path.exists():
            return EMPTY_OUTLINE
        return self._path.read_text(encoding='utf-8')

    def save(self, outline: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(outline, encoding='utf-8')
