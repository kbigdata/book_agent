"""
외부 API 오류 분류 모듈.

같은 분류를 두 곳에서 쓴다.
- graph.py  : 재시도 정책(retry_on). 다시 해도 실패할 오류(키/모델/잘못된 요청)는 재시도하지 않는다.
- app.py    : 사용자 안내. 설정 오류는 프로그램을 끝내고, 일시적 오류는 대화를 계속한다.
"""

from dataclasses import dataclass

import openai
from langgraph.errors import GraphRecursionError
from langgraph.types import default_retry_on
from tavily import errors as tavily_errors


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    message: str  # 사용자에게 보여줄 안내 문구
    fatal: bool = False  # 설정을 고치기 전에는 계속할 수 없다 -> 프로그램 종료
    retryable: bool = False  # 잠시 뒤 다시 하면 성공할 수 있다 -> 그래프가 재시도


def describe_error(exc: BaseException) -> ErrorInfo | None:
    """알려진 외부 API 오류면 ErrorInfo 를, 모르는 오류(코드 버그 등)면 None 을 돌려준다."""
    match exc:
        case openai.AuthenticationError():
            return ErrorInfo(
                'OpenAI API 키가 올바르지 않습니다. .env 의 OPENAI_API_KEY 를 확인하세요.',
                fatal=True,
            )
        case openai.PermissionDeniedError():
            return ErrorInfo(
                'OpenAI API 접근 권한이 없습니다. 키의 프로젝트/조직 권한을 확인하세요.', fatal=True
            )
        case openai.NotFoundError():
            return ErrorInfo(
                'OpenAI 모델을 찾을 수 없습니다. .env 의 LLM_MODEL 값을 확인하세요.', fatal=True
            )
        case (
            tavily_errors.InvalidAPIKeyError()
            | tavily_errors.MissingAPIKeyError()
            | tavily_errors.ForbiddenError()
        ):
            return ErrorInfo(
                'Tavily API 키가 올바르지 않습니다. .env 의 TAVILY_API_KEY 를 확인하세요.',
                fatal=True,
            )
        case openai.RateLimitError(code='insufficient_quota'):
            return ErrorInfo('OpenAI 크레딧이 부족합니다. 결제/사용 한도를 확인하세요.')
        case openai.RateLimitError():
            return ErrorInfo(
                'OpenAI 요청 한도를 넘었습니다. 잠시 후 다시 시도하세요.', retryable=True
            )
        case tavily_errors.UsageLimitExceededError():
            return ErrorInfo('Tavily 사용 한도를 넘었습니다. 남은 크레딧을 확인하세요.')
        case openai.APITimeoutError():  # APIConnectionError 의 하위 클래스라 먼저 검사한다
            # 클라이언트가 이미 LLM_MAX_RETRIES 만큼 다시 시도했다. 그래프까지 재시도하면
            # 멈춰 있는 시간이 몇 배로 늘어나므로 재시도하지 않는다.
            return ErrorInfo(
                'OpenAI 응답 시간이 초과되었습니다(LLM_TIMEOUT). 잠시 후 다시 시도하세요.'
            )
        case openai.APIConnectionError() | tavily_errors.TimeoutError():
            return ErrorInfo(
                '외부 API 서버에 연결하지 못했습니다. 네트워크를 확인하세요.', retryable=True
            )
        case openai.BadRequestError() | tavily_errors.BadRequestError():
            return ErrorInfo('외부 API 가 요청을 거부했습니다. 요청을 바꿔 다시 시도하세요.')
        case openai.APIStatusError(status_code=code) if code >= 500:
            return ErrorInfo(
                f'OpenAI 서버 오류입니다(HTTP {code}). 잠시 후 다시 시도하세요.', retryable=True
            )
        case openai.APIStatusError(status_code=code):
            return ErrorInfo(f'OpenAI 가 요청을 처리하지 못했습니다(HTTP {code}).')
        case GraphRecursionError():
            return ErrorInfo('작업 단계가 너무 많아 이번 요청을 중단했습니다. 요청을 나눠 주세요.')
    return None


def retry_on(exc: Exception) -> bool:
    """재시도 정책용 판정. 알려진 오류는 분류를 따르고, 나머지는 LangGraph 기본 규칙을 따른다."""
    info = describe_error(exc)
    return info.retryable if info is not None else default_retry_on(exc)
