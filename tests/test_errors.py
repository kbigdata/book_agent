"""오류 분류(errors.py) 시나리오. 실제 API 를 부르지 않고 예외 객체만 만들어 확인한다."""

import httpx
import openai
import pytest
from langgraph.errors import GraphRecursionError
from tavily import errors as tavily_errors

from book_agent.errors import describe_error, retry_on

REQUEST = httpx.Request('POST', 'https://api.openai.com/v1/chat/completions')


def status_error(cls, status, code=None):
    response = httpx.Response(status, request=REQUEST)
    return cls('error', response=response, body={'code': code} if code else None)


# (id, 예외, fatal, retryable, 안내 문구에 들어갈 단어)
CASES = [
    ('openai-401', status_error(openai.AuthenticationError, 401), True, False, 'OPENAI_API_KEY'),
    ('openai-403', status_error(openai.PermissionDeniedError, 403), True, False, '권한'),
    ('openai-404', status_error(openai.NotFoundError, 404), True, False, 'LLM_MODEL'),
    ('tavily-key', tavily_errors.InvalidAPIKeyError('bad'), True, False, 'TAVILY_API_KEY'),
    ('tavily-missing', tavily_errors.MissingAPIKeyError(), True, False, 'TAVILY_API_KEY'),
    ('tavily-403', tavily_errors.ForbiddenError('no'), True, False, 'TAVILY_API_KEY'),
    ('openai-429', status_error(openai.RateLimitError, 429), False, True, '한도'),
    (
        'openai-quota',
        status_error(openai.RateLimitError, 429, 'insufficient_quota'),
        False,
        False,
        '크레딧',
    ),
    ('tavily-limit', tavily_errors.UsageLimitExceededError('limit'), False, False, 'Tavily'),
    ('openai-conn', openai.APIConnectionError(request=REQUEST), False, True, '네트워크'),
    ('openai-timeout', openai.APITimeoutError(request=REQUEST), False, True, '네트워크'),
    ('tavily-timeout', tavily_errors.TimeoutError(30), False, True, '네트워크'),
    ('openai-400', status_error(openai.BadRequestError, 400), False, False, '거부'),
    ('tavily-400', tavily_errors.BadRequestError('bad'), False, False, '거부'),
    ('openai-500', status_error(openai.InternalServerError, 500), False, True, 'HTTP 500'),
    ('openai-503', status_error(openai.APIStatusError, 503), False, True, 'HTTP 503'),
    ('openai-409', status_error(openai.ConflictError, 409), False, False, 'HTTP 409'),
    ('recursion', GraphRecursionError('limit'), False, False, '단계'),
]


@pytest.mark.parametrize(
    'exc, fatal, retryable, keyword', [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_known_errors_are_classified(exc, fatal, retryable, keyword):
    """외부 API 오류마다 안내 문구, 종료 여부(fatal), 재시도 여부(retryable)가 정해져 있다.

    대상: errors.describe_error
    구분: 예외
    우선순위: 상
    사전조건: OpenAI/Tavily/LangGraph 예외 객체 생성(실제 호출 없음)
    절차: describe_error(exc)
    기대결과: 설정 오류(401/403/404/Tavily 키)는 fatal,
        일시적 오류(429/연결/타임아웃/5xx)는 retryable,
        크레딧 부족·400·기타 4xx 는 둘 다 아님, 안내 문구에 확인할 항목 포함
    """
    info = describe_error(exc)
    assert info is not None
    assert (info.fatal, info.retryable) == (fatal, retryable)
    assert keyword in info.message


@pytest.mark.parametrize('exc', [CASES[0][1], CASES[3][1]], ids=['openai-401', 'tavily-key'])
def test_error_message_hides_raw_detail(exc):
    """안내 문구에는 원본 오류 메시지(키 일부가 들어 있을 수 있음)를 넣지 않는다.

    대상: errors.describe_error
    구분: 예외
    우선순위: 중
    사전조건: 원본 메시지가 'error'/'bad' 인 인증 오류
    절차: describe_error(exc).message 확인
    기대결과: 원본 메시지 미포함
    """
    message = describe_error(exc).message
    assert str(exc) not in message


@pytest.mark.parametrize('exc', [ValueError('bug'), KeyError('x'), RuntimeError('boom')])
def test_unknown_errors_return_none(exc):
    """모르는 오류(코드 버그 등)는 분류하지 않아 앱이 traceback 을 남기게 한다.

    대상: errors.describe_error
    구분: 예외
    우선순위: 중
    사전조건: 일반 파이썬 예외
    절차: describe_error(exc)
    기대결과: None
    """
    assert describe_error(exc) is None


@pytest.mark.parametrize(
    'exc, expected',
    [
        (status_error(openai.AuthenticationError, 401), False),
        (status_error(openai.RateLimitError, 429), True),
        (openai.APIConnectionError(request=REQUEST), True),
        (status_error(openai.BadRequestError, 400), False),
        (ValueError('bug'), False),
        (ConnectionError('reset'), True),
        (Exception('generic'), True),
    ],
    ids=[
        'auth',
        'rate-limit',
        'connection',
        'bad-request',
        'value-error',
        'os-connection',
        'generic',
    ],
)
def test_retry_on(exc, expected):
    """재시도 판정: 분류된 오류는 retryable 을, 나머지는 LangGraph 기본 규칙을 따른다.

    대상: errors.retry_on
    구분: 정상
    우선순위: 상
    사전조건: 분류된 오류와 분류되지 않은 오류
    절차: retry_on(exc)
    기대결과: 인증·400 은 재시도 안 함, 429·연결은 재시도, ValueError 는 기본 규칙대로 재시도 안 함,
        ConnectionError·일반 Exception 은 기본 규칙대로 재시도
    """
    assert retry_on(exc) is expected
