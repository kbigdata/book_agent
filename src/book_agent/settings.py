'''
프로젝트 설정.

.env의 값은 두 경로로 쓰인다.
1. Settings 필드(OPEN_API_KEY 등) -> 타입 검증 후 코드에서 settings.xxx 로 사용
2. 라이브러리가 os.environ 에서 직접 읽는 값(HF_TOKEN, LANGSMITH_* 등)
-> pydantic-settings 는 os.environ에 넣어 주지 않으므로 load_dotenv() 로 따로 올린다.
'''

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/book_agent/settings.py -> 프로젝트 루트 이동
PROJECT_ROOT=Path(__file__).resolve().parents[2]
ENV_FILE=PROJECT_ROOT / '.env'

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file_encoding='utf-8', extra='ignore')

    openai_api_key:SecretStr # 필수: 없으면 시작 시점에 ValidationError
    llm_model:str='gpt-5.5'
    device:Literal['auto','cuda','cpu']='auto'
    data_dir: Path=PROJECT_ROOT / 'data'

def load_settings(env_file:Path=ENV_FILE)->Settings:
    load_dotenv(env_file, override=False)
    return Settings(_env_file=env_file)
