"""개발 환경 스모크 테스트. API 키 없이 실행된다."""

import os
import sys

import pytest

from book_agent.device import resolve_device
from book_agent.settings import load_settings


def test_python_version():
    """프로젝트가 요구하는 Python 버전에서 실행된다.

    대상: 실행 환경
    구분: 정상
    우선순위: 상
    사전조건: uv 가상환경(.venv)으로 pytest 실행
    절차: sys.version_info 의 major/minor 확인
    기대결과: (3, 12)
    """
    assert sys.version_info[:2] == (3, 12)


def test_settings_and_env_passthrough(tmp_path, monkeypatch):
    """.env 의 값이 Settings 필드와 os.environ 양쪽으로 전달된다.

    대상: settings.load_settings
    구분: 정상
    우선순위: 상
    사전조건: OPENAI_API_KEY, TAVILY_API_KEY, HF_TOKEN, LLM_MODEL 환경변수 제거
    절차: 1) 임시 .env 에 4개 값 기록 2) load_settings(.env 경로) 호출
    기대결과: 비밀값/모델명이 Settings 에 들어가고,
        Settings 에 없는 HF_TOKEN 은 os.environ 에 올라간다
    """
    for key in ('OPENAI_API_KEY', 'TAVILY_API_KEY', 'HF_TOKEN', 'LLM_MODEL'):
        monkeypatch.setenv(key, '')
        monkeypatch.delenv(key)
    env = tmp_path / '.env'
    env.write_text(
        'OPENAI_API_KEY=sk-test\nTAVILY_API_KEY=tvly-test\nLLM_MODEL=test-model\nHF_TOKEN=hf_test\n',
        encoding='utf-8',
    )

    settings = load_settings(env)

    assert settings.openai_api_key.get_secret_value() == 'sk-test'
    assert settings.llm_model == 'test-model'
    assert os.environ['HF_TOKEN'] == 'hf_test'  # Settings 에 없는 값도 os.environ에 올라감


def test_missing_key_fails_fast(tmp_path, monkeypatch):
    """필수 API 키가 없으면 시작 시점에 바로 실패한다.

    대상: settings.load_settings
    구분: 예외
    우선순위: 상
    사전조건: OPENAI_API_KEY 환경변수 제거
    절차: LLM_MODEL 만 있는 .env 로 load_settings 호출
    기대결과: openai_api_key 를 언급하는 ValidationError(ValueError)
    """
    # load_dotenv 가 os.environ 에 올리는 값도 테스트 후 원래대로 돌아가도록 등록한다.
    for key in ('OPENAI_API_KEY', 'LLM_MODEL'):
        monkeypatch.setenv(key, '')
        monkeypatch.delenv(key)
    (tmp_path / '.env').write_text('LLM_MODEL=x\n', encoding='utf-8')

    with pytest.raises(ValueError, match='openai_api_key'):
        load_settings(tmp_path / '.env')


def test_cuda_available():
    """GPU 가 있는 PC 에서는 auto 장치가 cuda 로 결정되고 실제 연산이 된다.

    대상: device.resolve_device
    구분: 정상
    우선순위: 하
    사전조건: torch 설치, CUDA 사용 가능(아니면 SKIP)
    절차: 1) resolve_device('auto') 2) cuda 텐서로 간단한 연산
    기대결과: 'cuda' 반환, 연산 결과 4
    """
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA를 사용할 수 없는 환경')
    assert resolve_device('auto') == 'cuda'
    assert (torch.ones(2, device='cuda') * 2).sum().item() == 4
