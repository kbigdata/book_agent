"""
프로젝트 설정.

.env의 값은 두 경로로 쓰인다.
1. Settings 필드(OPEN_API_KEY 등) -> 타입 검증 후 코드에서 settings.xxx 로 사용
2. 라이브러리가 os.environ 에서 직접 읽는 값(HF_TOKEN, LANGSMITH_* 등)
-> pydantic-settings 는 os.environ에 넣어 주지 않으므로 load_dotenv() 로 따로 올린다.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/book_agent/settings.py -> 프로젝트 루트 이동
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / '.env'


class Settings(BaseSettings):
    """
    모든 설정값을 한곳에 모은 타입 안전 설정 객체.

    환경변수 이름은 필드 이름의 대문자 형태다. 예) OPENAI_API_KEY, LLM_MODEL, DEVICE
    """

    model_config = SettingsConfigDict(env_file_encoding='utf-8', extra='ignore')

    # 비밀값 : SecretStr은 print/로그에 '*******'로 표시된다.
    openai_api_key: SecretStr  # 필수: 없으면 시작 시점에 ValidationError
    tavily_api_key: SecretStr

    # 모델
    llm_model: str = 'gpt-5.5'
    embedding_provider: Literal['openai', 'huggingface'] = 'openai'
    openai_embedding_model: str = 'text-embedding-3-large'
    hf_embedding_model: str = 'BAAI/bge-m3'

    device: Literal['auto', 'cuda', 'cpu'] = 'auto'

    # 외부 API 호출 제한: 응답이 없으면 llm_timeout 초 뒤 실패하고,
    # 클라이언트가 llm_max_retries 번 다시 시도한다.
    # 한 호출이 멈출 수 있는 최대 시간은 llm_timeout * (llm_max_retries + 1) 이다.
    llm_timeout: float = Field(default=120, gt=0)
    llm_max_retries: int = Field(default=1, ge=0)

    # 저장 위치
    data_dir: Path = PROJECT_ROOT / 'data'

    # 동적 파라미터
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    search_top_k: int = Field(default=5, gt=0)
    max_research_rounds: int = Field(default=2, ge=0)
    max_tool_calls: int = Field(default=6, gt=0)
    max_history_tokens: int = Field(default=8000, gt=0)
    thread_id: str = 'book_001'

    @property
    def outline_path(self) -> Path:
        return self.data_dir / 'outline.md'

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / 'chroma'

    @property
    def resources_dir(self) -> Path:
        return self.data_dir / 'resources'

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / 'checkpoints.sqlite'


def load_settings(env_file: Path = ENV_FILE) -> Settings:
    load_dotenv(env_file, override=False)
    return Settings(_env_file=env_file)
