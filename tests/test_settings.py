"""설정(settings.py)과 장치 선택(device.py) 시나리오."""

import sys

import pytest
from pydantic import ValidationError

from book_agent.device import resolve_device


def test_defaults(make_settings):
    """필수 키만 주면 나머지 설정은 문서화된 기본값을 갖는다.

    대상: settings.Settings
    구분: 정상
    우선순위: 중
    사전조건: 설정 관련 환경변수 제거, .env 미사용
    절차: API 키 두 개만 넣어 Settings 생성
    기대결과: llm_model=gpt-5.5, embedding_provider=openai, chunk 1000/100, top_k 5, 조사 한도 2,
        도구 호출 한도 6, thread_id=book_001, LLM 타임아웃 120초·클라이언트 재시도 1회
    """
    s = make_settings()
    assert s.llm_model == 'gpt-5.5'
    assert s.embedding_provider == 'openai'
    assert s.device == 'auto'
    assert (s.chunk_size, s.chunk_overlap, s.search_top_k) == (1000, 100, 5)
    assert (s.max_research_rounds, s.max_tool_calls, s.max_history_tokens) == (2, 6, 8000)
    assert s.thread_id == 'book_001'
    assert (s.llm_timeout, s.llm_max_retries) == (120, 1)


def test_derived_paths_follow_data_dir(make_settings, tmp_path):
    """저장 경로는 모두 data_dir 아래에 만들어진다.

    대상: settings.Settings 경로 속성
    구분: 정상
    우선순위: 중
    사전조건: data_dir=tmp_path
    절차: outline_path, chroma_dir, resources_dir, checkpoint_db 조회
    기대결과: 각각 outline.md, chroma, resources, checkpoints.sqlite
    """
    s = make_settings()
    assert s.outline_path == tmp_path / 'outline.md'
    assert s.chroma_dir == tmp_path / 'chroma'
    assert s.resources_dir == tmp_path / 'resources'
    assert s.checkpoint_db == tmp_path / 'checkpoints.sqlite'


def test_secrets_are_masked(make_settings):
    """API 키는 출력/로그에 원문이 드러나지 않는다.

    대상: settings.Settings (SecretStr)
    구분: 정상
    우선순위: 상
    사전조건: openai_api_key='sk-secret-value'
    절차: repr(settings), str(settings.openai_api_key) 확인
    기대결과: 원문 미노출('**********'), get_secret_value() 로만 원문 획득
    """
    s = make_settings(openai_api_key='sk-secret-value')
    assert 'sk-secret-value' not in repr(s)
    assert str(s.openai_api_key) == '**********'
    assert s.openai_api_key.get_secret_value() == 'sk-secret-value'


def test_env_vars_override_defaults(make_settings, monkeypatch):
    """환경변수 값이 기본값보다 우선하고 타입이 변환된다.

    대상: settings.Settings
    구분: 정상
    우선순위: 중
    사전조건: LLM_MODEL=custom, SEARCH_TOP_K=9, EMBEDDING_PROVIDER=huggingface 설정
    절차: Settings 생성
    기대결과: 문자열 '9' 가 int 9 로 변환되어 적용
    """
    monkeypatch.setenv('LLM_MODEL', 'custom')
    monkeypatch.setenv('SEARCH_TOP_K', '9')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'huggingface')
    s = make_settings()
    assert (s.llm_model, s.search_top_k, s.embedding_provider) == ('custom', 9, 'huggingface')


def test_missing_tavily_key_fails(make_settings):
    """Tavily 키가 없으면 설정 생성이 실패한다.

    대상: settings.Settings
    구분: 예외
    우선순위: 상
    사전조건: TAVILY_API_KEY 환경변수 없음
    절차: tavily_api_key 없이 Settings 생성
    기대결과: tavily_api_key 를 언급하는 ValidationError
    """
    from book_agent.settings import Settings

    with pytest.raises(ValidationError, match='tavily_api_key'):
        Settings(openai_api_key='sk', _env_file=None)


@pytest.mark.parametrize(
    'field, value',
    [
        ('chunk_size', 0),
        ('chunk_overlap', -1),
        ('search_top_k', 0),
        ('max_research_rounds', -1),
        ('max_tool_calls', 0),
        ('max_history_tokens', 0),
        ('llm_timeout', 0),
        ('llm_max_retries', -1),
        ('embedding_provider', 'cohere'),
        ('device', 'mps'),
    ],
)
def test_invalid_values_rejected(make_settings, field, value):
    """범위를 벗어나거나 허용 목록에 없는 설정값은 거부된다.

    대상: settings.Settings 필드 검증
    구분: 경계
    우선순위: 중
    사전조건: 없음
    절차: 해당 필드에 잘못된 값을 넣어 Settings 생성
    기대결과: 해당 필드명을 언급하는 ValidationError
    """
    with pytest.raises(ValidationError, match=field):
        make_settings(**{field: value})


def test_zero_research_rounds_allowed(make_settings):
    """조사 한도 0 은 '조사하지 않음'으로 허용된다(ge=0 경계).

    대상: settings.Settings.max_research_rounds
    구분: 경계
    우선순위: 하
    사전조건: 없음
    절차: max_research_rounds=0 으로 생성
    기대결과: 오류 없이 0 저장
    """
    assert make_settings(max_research_rounds=0).max_research_rounds == 0


@pytest.mark.parametrize('preference', ['cpu', 'cuda'])
def test_explicit_device_is_returned_as_is(preference):
    """auto 가 아닌 장치는 확인 없이 그대로 사용한다.

    대상: device.resolve_device
    구분: 정상
    우선순위: 중
    사전조건: 없음
    절차: resolve_device(cpu/cuda)
    기대결과: 입력값 그대로 반환
    """
    assert resolve_device(preference) == preference


def test_auto_device_without_torch_is_cpu(monkeypatch):
    """torch 가 없는 PC(--no-group gpu)에서는 auto 가 cpu 로 결정된다.

    대상: device.resolve_device
    구분: 예외
    우선순위: 중
    사전조건: sys.modules['torch']=None 으로 import 실패 유도
    절차: resolve_device('auto')
    기대결과: 'cpu'
    """
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert resolve_device('auto') == 'cpu'


def test_auto_device_matches_cuda_availability():
    """auto 는 torch.cuda.is_available() 결과를 따른다.

    대상: device.resolve_device
    구분: 정상
    우선순위: 하
    사전조건: torch 설치(아니면 SKIP)
    절차: resolve_device('auto') 와 torch.cuda.is_available() 비교
    기대결과: 사용 가능하면 cuda, 아니면 cpu
    """
    torch = pytest.importorskip('torch')
    assert resolve_device('auto') == ('cuda' if torch.cuda.is_available() else 'cpu')
