"""
실제 API 통합 시나리오. `pytest --live` 로만 실행된다(OpenAI/Tavily 비용 발생).

.env 의 키와 모델 설정을 그대로 쓰고, 저장 위치(data_dir)만 임시 폴더로 바꿔
실제 data/ 폴더의 목차·벡터 DB·체크포인트는 건드리지 않는다.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tavily import TavilyClient

from book_agent.app import build_context, run_turn
from book_agent.graph import build_graph
from book_agent.models import WORKERS, TaskPlan
from book_agent.nodes import _now
from book_agent.prompts import RESEARCH_REQUEST, SUPERVISOR_PROMPT
from book_agent.services import VectorStoreService, WebSearchService, build_embeddings
from book_agent.settings import load_settings

pytestmark = pytest.mark.live


@pytest.fixture(scope='module')
def settings(tmp_path_factory):
    real = load_settings()
    return real.model_copy(update={'data_dir': tmp_path_factory.mktemp('live_data')})


@pytest.fixture(scope='module')
def ctx(settings):
    return build_context(settings)


def config(thread):
    return {'configurable': {'thread_id': thread}, 'recursion_limit': 40}


def test_env_settings_loaded(settings):
    """.env 의 실제 키와 모델 설정이 올바르게 읽힌다(값의 주석은 제거).

    대상: settings.load_settings (실제 .env)
    구분: 정상
    우선순위: 상
    사전조건: 프로젝트 루트 .env 에 OPENAI_API_KEY, TAVILY_API_KEY 설정
    절차: load_settings() 후 키 존재와 설정값 확인(키 원문은 출력하지 않음)
    기대결과: 두 키가 비어 있지 않고, embedding_provider/device 가 허용 값으로 파싱됨
    """
    assert settings.openai_api_key.get_secret_value()
    assert settings.tavily_api_key.get_secret_value()
    assert settings.embedding_provider in ('openai', 'huggingface')
    assert settings.device in ('auto', 'cuda', 'cpu')


def test_openai_chat_responds(ctx, settings):
    """설정된 LLM 모델이 실제로 응답한다.

    대상: ChatOpenAI (settings.llm_model)
    구분: 정상
    우선순위: 상
    사전조건: 유효한 OPENAI_API_KEY
    절차: '대한민국의 수도를 한 단어로' 질문
    기대결과: 비어 있지 않은 AIMessage, '서울' 포함
    """
    reply = ctx.llm.invoke('대한민국의 수도를 한 단어로만 답해.')
    assert isinstance(reply, AIMessage)
    assert '서울' in reply.text


def test_planner_structured_output(ctx):
    """supervisor 플래너가 strict JSON 스키마로 TaskPlan 을 돌려준다.

    대상: llm.with_structured_output(TaskPlan, json_schema, strict)
    구분: 정상
    우선순위: 상
    사전조건: 목차·노트 없음, 사용자 요청 1건
    절차: SUPERVISOR_PROMPT | planner 실행
    기대결과: TaskPlan 인스턴스, agent 는 WORKERS 중 하나, description 비어 있지 않음
    """
    plan = (SUPERVISOR_PROMPT | ctx.planner).invoke(
        {
            'now': _now(),
            'outline': '아직 작성된 목차가 없습니다.',
            'research_notes': '아직 조사 노트가 없습니다.',
            'task_log': '아직 수행한 작업이 없습니다.',
            'research_rounds': 0,
            'max_research_rounds': 2,
            'messages': [HumanMessage('파이썬 입문서 목차를 만들어 줘.')],
        }
    )
    assert isinstance(plan, TaskPlan)
    assert plan.agent in WORKERS
    assert plan.description.strip()


def test_embeddings_vector(settings):
    """설정된 임베딩 모델이 실제 벡터를 만든다.

    대상: services.build_embeddings (실제 모델)
    구분: 정상
    우선순위: 상
    사전조건: embedding_provider 설정(.env)
    절차: embed_query 2회(같은 문장)
    기대결과: 길이 > 0 인 실수 벡터, 같은 문장은 거의 같은 벡터(코사인 > 0.99)
    """
    embeddings = build_embeddings(settings)
    a = embeddings.embed_query('파이썬 입문서')
    b = embeddings.embed_query('파이썬 입문서')
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    assert len(a) > 0
    assert dot / norm > 0.99


def test_tavily_search(settings):
    """Tavily 실제 검색 결과가 Document 로 변환되고 JSON 으로 저장된다.

    대상: services.WebSearchService + TavilyClient
    구분: 정상
    우선순위: 상
    사전조건: 유효한 TAVILY_API_KEY, max_results=2
    절차: search('Python 3.13 new features')
    기대결과: 1~2건, 각 문서의 source 가 http 로 시작, 본문 비어 있지 않음, JSON 1개 저장
    """
    save_dir = settings.resources_dir / 'tavily_test'
    web = WebSearchService(
        TavilyClient(api_key=settings.tavily_api_key.get_secret_value()), save_dir, max_results=2
    )
    docs = web.search('Python 3.13 new features')

    assert 1 <= len(docs) <= 2
    for d in docs:
        assert d.metadata['source'].startswith('http')
        assert d.page_content.strip()
    [saved] = save_dir.glob('*.json')
    assert json.loads(saved.read_text(encoding='utf-8'))['query'] == 'Python 3.13 new features'


def test_vector_search_with_real_embeddings(settings):
    """실제 임베딩으로 저장한 문서 중 의미가 가까운 문서가 먼저 검색된다.

    대상: services.VectorStoreService (실제 임베딩)
    구분: 정상
    우선순위: 상
    사전조건: 주제가 다른 문서 3건(파이썬/요리/여행)
    절차: '프로그래밍 언어 문법' 으로 검색(k=1)
    기대결과: 파이썬 문서가 1순위
    """
    from langchain_core.documents import Document

    vectors = VectorStoreService(
        persist_dir=settings.chroma_dir,
        embeddings=build_embeddings(settings),
        collection_name='live_test',
    )
    vectors.add_documents(
        [
            Document('파이썬의 변수, 조건문, 반복문 문법', metadata={'source': 'py', 'title': 'a'}),
            Document('김치찌개를 맛있게 끓이는 방법', metadata={'source': 'food', 'title': 'b'}),
            Document('제주도 3박 4일 여행 코스', metadata={'source': 'trip', 'title': 'c'}),
        ]
    )
    [top] = vectors.search('프로그래밍 언어 문법', k=1)
    assert top.metadata['source'] == 'py'


def test_researcher_agent_uses_tools(ctx):
    """researcher 하위 에이전트가 실제 도구를 써서 출처가 붙은 요약을 만든다.

    대상: app.build_context 의 researcher (create_agent + tools)
    구분: 정상
    우선순위: 상
    사전조건: 빈 벡터 DB, 도구 호출 한도(max_tool_calls)
    절차: RESEARCH_REQUEST 로 조사 임무 1건 실행
    기대결과: 도구 호출 1회 이상·한도 이하, 최종 답변 비어 있지 않고 URL(http) 포함,
        extract_sources 결과가 1건 이상이고 모두 도구 출력에 있던 URL·제목
    """
    request = RESEARCH_REQUEST.format(
        mission='Python 3.13 의 주요 신기능 3가지를 조사',
        now=_now(),
        outline='아직 작성된 목차가 없습니다.',
        known='아직 조사 노트가 없습니다.',
    )
    result = ctx.researcher.invoke({'messages': [HumanMessage(request)]})

    tool_calls = [c for m in result['messages'] for c in getattr(m, 'tool_calls', [])]
    assert len(tool_calls) >= 1
    final = result['messages'][-1].text
    assert final.strip()
    assert 'http' in final

    # 출처 구조화: 답변이 인용한 URL 중 도구가 실제로 돌려준 것만 Source 로 남는다.
    from langchain_core.messages import ToolMessage

    from book_agent.tools import extract_sources

    sources = extract_sources(result['messages'])
    tool_text = '\n'.join(m.text for m in result['messages'] if isinstance(m, ToolMessage))
    assert sources, '실제 조사 결과에서 출처를 하나도 추출하지 못했다'
    assert all(s.url in tool_text and s.title for s in sources)


def test_full_turn_with_real_apis(ctx, capsys):
    """실제 API 로 한 턴을 실행하면 목차가 저장되고 human 에서 다음 지시를 기다린다.

    대상: graph 전체 흐름 (실제 LLM/검색/임베딩)
    구분: 정상
    우선순위: 상
    사전조건: 빈 data_dir, 구체적인 요청(조사 1회, 5장 목차)
    절차: run_turn 으로 1턴 실행
    기대결과: next=('human',), 모든 작업 done, 마지막 메시지는 AI 보고, 목차 파일에 '#' 제목 포함
    """
    graph = build_graph(checkpointer=InMemorySaver())
    request = (
        '초보자용 파이썬 입문서 목차를 5장 정도로 만들어 줘. '
        '최신 파이썬 버전 정보만 한 번 조사하고, 질문 없이 바로 목차를 작성해.'
    )

    run_turn(graph, {'messages': [HumanMessage(request)]}, config('live-1'), ctx)

    state = graph.get_state(config('live-1'))
    print(capsys.readouterr().out)  # -s 로 실행하면 진행 로그 확인 가능
    assert state.next == ('human',)
    assert all(t.status == 'done' for t in state.values['tasks'])
    assert isinstance(state.values['messages'][-1], AIMessage)
    assert '#' in ctx.outline_repo.load()


def test_resume_edits_outline_with_real_apis(ctx):
    """이전 턴에 이어서 수정 지시를 주면 목차가 갱신되고 다시 대기한다.

    대상: graph 재개 (Command(resume)) + content_strategist
    구분: 정상
    우선순위: 상
    사전조건: 직전 시나리오로 목차가 저장되고 human 에서 대기 중
    절차: '마지막에 6장 「실전 프로젝트」를 추가해 줘' 로 재개
    기대결과: 다시 human 대기, 목차에 '실전 프로젝트' 포함, 사용자 메시지가 대화에 추가
    """
    graph = build_graph(checkpointer=InMemorySaver())
    request = '초보자용 파이썬 입문서 목차를 4장으로 바로 만들어 줘. 조사하지 말고 질문도 하지 마.'
    graph.invoke({'messages': [HumanMessage(request)]}, config('live-2'), context=ctx)
    assert graph.get_state(config('live-2')).next == ('human',)

    graph.invoke(
        Command(resume='마지막에 「실전 프로젝트」 장을 하나 추가해 줘. 질문 없이 바로 반영해.'),
        config('live-2'),
        context=ctx,
    )

    state = graph.get_state(config('live-2'))
    assert state.next == ('human',)
    assert '실전 프로젝트' in ctx.outline_repo.load()
    assert any('실전 프로젝트' in m.text for m in state.values['messages'] if m.type == 'human')
