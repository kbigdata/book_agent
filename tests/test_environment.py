'''개발 환경 스모크 테스트. API 키 없이 실행된다.'''

import os
import sys

import pytest
from book_agent.device import resolve_device
from book_agent.settings import load_settings

def test_python_version():
    assert sys.version_info[:2] == (3,12)

def test_settings_and_env_passthrough(tmp_path, monkeypatch):
    for key in ('OPENAI_API_KEY','HF_TOKEN','LLM_MODEL'):
        monkeypatch.setenv(key,'')
        monkeypatch.delenv(key)
    env=tmp_path / '.env'
    env.write_text(
        'OPENAI_API_KEY=sk-test\nLLM_MODEL=test-model\nHF_TOKEN=hf_test\n', encoding='utf-8'
    )

    settings=load_settings(env)

    assert settings.openai_api_key.get_secret_value() == 'sk-test'
    assert settings.llm_model=='test-model'
    assert os.environ['HF_TOKEN'] == 'hf_test' # Settings 에 없는 값도 os.environ에 올라감

def test_missing_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API-KEY', raising=False)
    (tmp_path / '.env').write_text('LLM_MODEL=x\n', encoding='utf-8')

    with pytest.raises(ValueError, match='openai_api_key'):
        load_settings(tmp_path / '.env')

def test_cuda_available():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA를 사용할 수 없는 환경')
    assert resolve_device('auto') == 'cuda'
    assert (torch.ones(2, device='cuda') * 2).sum().item() == 4
    