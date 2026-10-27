"""
애플리케이션 진입점(Composition Root).

설정을 읽고 -> 의존성을 조립하고 -> 체크포인트와 그래프를 만들고 -> 대화 루프를 돈다.
객체 생성은 모두 이 파일에서만 일어나므로 나머지 모듈은 테스트하기 쉽다.
"""

import logging
import sqlite3
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError
from tavily import TavilyClient

from .context import AppContext
from .errors import ErrorInfo, describe_error, retry_on
from .graph import build_graph
from .models import TaskPlan
from .prompts import RESEARCHER_SYSTEM_PROMPT
from .repositories import FileOutlineRepository
from .services import VectorStoreService, WebSearchService, build_embeddings
from .settings import Settings, load_settings
from .tools import build_research_tools

logger = logging.getLogger(__name__)

EXIT_WORDS = {'q', 'quit', 'exit'}
STREAM_NODES = {'content_strategist', 'communicator'}  # 토큰 단위로 출력할 노드

# 체크포인트에서 복원을 허용할 사용자 정의 타입 (이 외의 타입은 역직렬화하지 않는다).
CHECKPOINT_TYPES = [
    ('book_agent.models', 'Task'),
    ('book_agent.models', 'ResearchNote'),
    ('book_agent.models', 'Source'),  # ResearchNote.sources 안의 타입도 등록해야 복원된다
]


def build_context(settings: Settings) -> AppContext:
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,  # 기본값(None)이면 응답이 없을 때 끝없이 기다린다
        max_retries=settings.llm_max_retries,
    )

    vectors = VectorStoreService(
        persist_dir=settings.chroma_dir,
        embeddings=build_embeddings(settings),
        # 임베딩 모델마다 벡터 차원이 다르므로 컬렉션을 분리한다.
        collection_name=f'web_{settings.embedding_provider}',
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    web = WebSearchService(
        client=TavilyClient(api_key=settings.tavily_api_key.get_secret_value()),
        save_dir=settings.resources_dir,
    )

    researcher = create_agent(
        model=llm,
        tools=build_research_tools(web, vectors, top_k=settings.search_top_k),
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        middleware=[
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls),  # 검색 비용 상한
            # 일시적 API 오류만 재시도하고, 끝내 실패하면 오류를 올려 그래프/앱이 처리하게 한다.
            # (기본값 'continue' 는 오류 문구를 조사 요약으로 저장해 버린다.)
            ModelRetryMiddleware(max_retries=2, retry_on=retry_on, on_failure='error'),
        ],
        name='researcher',
    )

    return AppContext(
        llm=llm,
        planner=llm.with_structured_output(TaskPlan, method='json_schema', strict=True),
        researcher=researcher,
        outline_repo=FileOutlineRepository(settings.outline_path),
        max_research_rounds=settings.max_research_rounds,
        max_history_tokens=settings.max_history_tokens,
    )


@contextmanager
def open_checkpointer(db_path: Path) -> Generator[SqliteSaver]:
    """SQLite 체크포인터를 열고, with 블록이 끝나면 연결을 닫는다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # LangGraph 는 다른 스레드에서 체크포인트를 쓸 수 있으므로 스레드 검사를 끈다.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
        yield SqliteSaver(conn, serde=serde)
    finally:
        conn.close()


def run_turn(
    graph: CompiledStateGraph, payload: dict | Command, config: RunnableConfig, ctx: AppContext
) -> None:
    """한 턴을 실행하며 노드 진행 상황(updates)과 LLM 토큰(messages)을 출력한다."""
    speaking = None
    for mode, chunk in graph.stream(
        payload, config, context=ctx, stream_mode=['updates', 'messages']
    ):
        if mode == 'messages':
            token, meta = chunk
            node = meta.get('langgraph_node')
            # supervisor 의 JSON 출력이나 researcher 의 도구 호출은 화면에 흘리지 않는다.
            if node in STREAM_NODES and token.text:
                if node != speaking:
                    print(f'\n\n[{node}]')
                    speaking = node
                print(token.text, end='', flush=True)
        else:  # 'updates': {노드 이름: 그 노드가 반환한 변경분}
            for node, update in chunk.items():
                if node == 'supervisor':
                    task = update['tasks'][-1]
                    print(f'\n[supervisor] 다음 작업 -> {task.agent}: {task.description}')
                elif node == 'researcher':
                    sources = len(update['research_notes'][-1].sources)
                    rounds = update['research_rounds']
                    print(f'\n[researcher] 조사 완료 ({rounds}회차, 출처 {sources}건)')
    print()


def waiting_for_human(graph: CompiledStateGraph, config: RunnableConfig) -> bool:
    """
    human 노드의 interrupt 가 입력을 기다리는 중인지 확인한다.

    state.next 만 보면 안 된다. 턴 도중 오류로 멈춘 경우에도 next 가 남는데,
    그때 resume 하면 사용자의 새 입력이 버려진다. 그 경우는 새 메시지로 다시 시작한다.
    """
    return bool(graph.get_state(config).interrupts)


def use_utf8_io() -> None:
    """
    표준입출력을 UTF-8 로 고정한다.

    Windows 에서 입출력을 파이프/파일로 연결하면 기본값이 cp949 인데,
    LLM 답변에 자주 나오는 em dash(—)·이모지는 cp949 에 없어 UnicodeEncodeError 가 난다.
    입력도 함께 바꿔야 같은 인코딩으로 한글을 주고받을 수 있다.
    잘못된 바이트가 들어와도 멈추지 않도록 errors='replace' 를 둔다.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


def main() -> None:
    use_utf8_io()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    try:
        settings = load_settings()
    except ValidationError as e:
        names = ', '.join(sorted({str(err['loc'][0]).upper() for err in e.errors()}))
        sys.exit(f'[설정 오류] .env 의 다음 값을 확인하세요: {names}')
    ctx = build_context(settings)
    config: RunnableConfig = {
        'configurable': {'thread_id': settings.thread_id},
        'recursion_limit': 40,  # 노드 이동 횟수 상한 (무한 루프 방지)
    }

    with open_checkpointer(settings.checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        print(f'AI 목차 작성 도우미 (thread: {settings.thread_id}, 목차: {settings.outline_path})')
        print("종료: 'q' - 같은 thread_id 로 다시 실행하면 이전 대화를 이어 간다.")

        while True:
            try:
                user_input = input('\nUser\t: ').strip()
            except (EOFError, KeyboardInterrupt):  # 입력 스트림 종료, Ctrl+C
                print('\nGoodbye!!')
                break
            if not user_input:
                continue
            if user_input.lower() in EXIT_WORDS:
                print('Goodbye!!')
                break

            # human 노드에서 멈춰 있으면 재개(resume)하고, 아니면 새 메시지로 시작한다.
            payload = (
                Command(resume=user_input)
                if waiting_for_human(graph, config)
                else {'messages': [HumanMessage(user_input)]}
            )
            try:
                run_turn(graph, payload, config, ctx)
            except KeyboardInterrupt:
                print('\n[중단] 이번 요청을 멈췄습니다. 다음 지시를 입력하세요.')
            except Exception as exc:
                info = describe_error(exc)
                if info is None:  # 모르는 오류는 코드 버그일 수 있으므로 traceback 을 남긴다.
                    logger.exception('처리하지 못한 오류')
                    info = ErrorInfo(f'예기치 못한 오류가 발생했습니다: {type(exc).__name__}')
                print(f'\n[오류] {info.message}', file=sys.stderr)
                if info.fatal:
                    sys.exit(1)
                print('대화 기록은 저장되어 있습니다. 다음 지시를 입력하면 이어서 진행합니다.')


if __name__ == '__main__':
    main()
