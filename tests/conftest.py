"""
공통 픽스처와 시나리오 결과 수집 훅.

모든 테스트는 API 키와 네트워크 없이 실행된다.
LLM, 플래너, 하위 에이전트는 가짜 객체로 바꾸고, 파일은 tmp_path 아래에만 만든다.

`pytest --scenario-report=PATH` 로 실행하면 테스트마다 docstring(시나리오)과 결과를
JSON 으로 남긴다. scripts/build_test_report.py 가 이 파일로 엑셀 보고서를 만든다.
"""

import inspect
import json
import platform
import sys
from collections.abc import Callable, Iterable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from book_agent.context import AppContext
from book_agent.models import TaskPlan, Worker
from book_agent.repositories import FileOutlineRepository
from book_agent.settings import Settings


# ----------------------------------------------------------------- 가짜 객체 / 픽스처
class FakeSearchClient:
    """TavilyClient 대역. 호출 인자를 기록하고 정해진 결과를 돌려준다."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({'query': query, **kwargs})
        return {'results': self.results}


def plan(agent: Worker, description: str = '작업') -> TaskPlan:
    return TaskPlan(agent=agent, description=description)


def capture_prompt(seen: dict[str, Any], result: Any) -> RunnableLambda:
    """프롬프트로 만들어진 메시지를 seen['messages'] 에 기록하고 result 를 돌려주는 대역."""

    def run(prompt_value: Any) -> Any:
        seen['messages'] = prompt_value.to_messages()
        return result

    return RunnableLambda(run)


@pytest.fixture
def make_ctx(tmp_path: Path) -> Callable[..., AppContext]:
    """
    가짜 의존성으로 AppContext 를 만드는 팩토리.

    plans     : supervisor 가 차례로 내놓을 TaskPlan 목록
    replies   : LLM(content_strategist/communicator)이 차례로 내놓을 답변
    summary   : researcher 하위 에이전트의 최종 답변
    """

    def factory(
        plans: Iterable[TaskPlan] = (),
        replies: list[str] | None = None,
        summary: str = '- 조사한 사실 (https://example.com)',
        **overrides: Any,
    ) -> AppContext:
        queue = iter(plans)
        options: dict[str, Any] = {
            'llm': FakeListChatModel(responses=replies or ['응답']),
            'planner': RunnableLambda(lambda _: next(queue)),
            'researcher': RunnableLambda(lambda _: {'messages': [AIMessage(summary)]}),
            'outline_repo': FileOutlineRepository(tmp_path / 'outline.md'),
        }
        options.update(overrides)
        return AppContext(**options)

    return factory


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """재시도 대기 시간만 없앤 정책으로 바꾼다(횟수와 retry_on 판정은 실제 정책과 같다)."""
    from langgraph.types import RetryPolicy

    import book_agent.graph as graph_module

    real = graph_module.LLM_RETRY
    monkeypatch.setattr(
        graph_module,
        'LLM_RETRY',
        RetryPolicy(
            max_attempts=real.max_attempts,
            initial_interval=0.0,
            jitter=False,
            retry_on=real.retry_on,
        ),
    )


@pytest.fixture
def make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """셸 환경변수와 .env 의 영향을 지운 Settings 팩토리 (data_dir 은 tmp_path)."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

    def factory(**overrides: Any) -> Settings:
        options: dict[str, Any] = {
            'openai_api_key': 'sk-test',
            'tavily_api_key': 'tvly-test',
            'data_dir': tmp_path,
            '_env_file': None,
        }
        options.update(overrides)
        return Settings(**options)

    return factory


# ----------------------------------------------------------------- 시나리오 결과 수집
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--scenario-report',
        metavar='PATH',
        help='테스트 시나리오와 실행 결과를 JSON 으로 저장한다.',
    )
    parser.addoption(
        '--live',
        action='store_true',
        help='.env 의 실제 API 키로 OpenAI/Tavily 를 호출하는 live 테스트도 실행한다(비용 발생).',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', 'live: 실제 외부 API 를 호출하는 테스트 (--live 로 실행)')


_scenarios: dict[str, dict[str, Any]] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption('--live'):
        skip_live = pytest.mark.skip(reason='실제 API 호출 테스트: --live 옵션으로 실행')
        for item in items:
            if 'live' in item.keywords:
                item.add_marker(skip_live)
    for item in items:
        if not isinstance(item, pytest.Function):
            continue
        callspec = getattr(item, 'callspec', None)
        _scenarios[item.nodeid] = {
            'nodeid': item.nodeid,
            'file': item.path.name,
            'function': item.originalname,
            'param': callspec.id if callspec else '',
            'doc': inspect.getdoc(item.function) or '',
            'outcome': None,
            'duration': 0.0,
            'message': '',
        }


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    record = _scenarios.get(report.nodeid)
    if record is None:
        return
    record['duration'] += report.duration
    if report.failed:
        record['outcome'] = 'failed' if report.when == 'call' else 'error'
        record['message'] = report.longreprtext[-1500:]
    elif report.skipped:
        record['outcome'] = 'skipped'
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else report.longrepr
        record['message'] = str(reason).removeprefix('Skipped: ')
    elif report.when == 'call' and record['outcome'] is None:
        record['outcome'] = 'passed'


_started = datetime.now()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = session.config.getoption('--scenario-report')
    if not path:
        return
    packages = [
        'pytest',
        'langchain',
        'langchain-core',
        'langgraph',
        'langchain-chroma',
        'pydantic',
    ]
    try:
        import torch

        device = f'torch {torch.__version__}, CUDA 사용 가능: {torch.cuda.is_available()}'
    except ImportError:
        device = 'torch 미설치'

    payload = {
        'started_at': _started.isoformat(timespec='seconds'),
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'exit_status': int(exitstatus),
        'command': 'pytest ' + ' '.join(session.config.invocation_params.args),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'device': device,
        'packages': {name: metadata.version(name) for name in packages},
        'results': [r for r in _scenarios.values() if r['outcome'] is not None],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
