"""출처 구조화 시나리오: 도구 출력에서 출처 추출 → 조사 노트 저장 → 체크포인트 복원 → 진행 표시."""

import pytest
from conftest import plan
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from book_agent.app import open_checkpointer, run_turn
from book_agent.graph import build_graph
from book_agent.models import ResearchNote, Source, Task
from book_agent.nodes import ResearcherNode
from book_agent.tools import extract_sources

WEB_OUTPUT = (
    '2건 검색, 새 청크 2개 저장\n'
    ' - 파이썬 공식 문서 (https://docs.python.org)\n'
    '- PEP 8 (https://peps.python.org/pep-0008)'
)
RETRIEVE_OUTPUT = (
    "[1] What's New In Python 3.13 (https://docs.python.org/3/whatsnew/3.13.html)\n본문 발췌"
)


def tool(text: str, call_id: str = '1') -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call_id)


def agent_returning(*messages) -> RunnableLambda:
    return RunnableLambda(lambda _: {'messages': list(messages)})


def config(thread='t1'):
    return {'configurable': {'thread_id': thread}, 'recursion_limit': 40}


# ------------------------------------------------------------------ extract_sources
def test_extract_sources_keeps_cited_tool_urls():
    """답변에 인용된 URL 중 도구 출력에 있는 것만 제목과 함께 출처가 된다.

    대상: tools.extract_sources
    구분: 정상
    우선순위: 상
    사전조건: web_search·retrieve 도구 출력 2건, 최종 답변이 URL 2개를 인용(문장 끝 마침표 포함)
    절차: extract_sources(messages)
    기대결과: 인용 순서대로 Source 2건, 제목은 도구 출력의 제목, URL 끝 마침표 제거
    """
    answer = AIMessage(
        '- 3.13 에서 대화형 인터프리터가 개선됐다 (https://docs.python.org/3/whatsnew/3.13.html)\n'
        '- 코드 스타일은 PEP 8 을 따른다 (https://peps.python.org/pep-0008).'
    )
    sources = extract_sources([tool(WEB_OUTPUT), tool(RETRIEVE_OUTPUT, '2'), answer])
    assert [(s.title, s.url) for s in sources] == [
        ("What's New In Python 3.13", 'https://docs.python.org/3/whatsnew/3.13.html'),
        ('PEP 8', 'https://peps.python.org/pep-0008'),
    ]


def test_extract_sources_drops_invented_urls():
    """도구가 돌려주지 않은 URL(LLM 이 지어낸 출처)은 버린다.

    대상: tools.extract_sources
    구분: 예외
    우선순위: 상
    사전조건: 도구 출력에는 docs.python.org 가 있음, 답변은 존재하지 않는 URL 도 인용
    절차: extract_sources(messages)
    기대결과: 도구 출력에 있던 URL 1건만 남음
    """
    answer = AIMessage(
        '- 사실 A (https://docs.python.org)\n- 사실 B (https://made-up.example.com/fake)'
    )
    assert [s.url for s in extract_sources([tool(WEB_OUTPUT), answer])] == [
        'https://docs.python.org'
    ]


def test_extract_sources_dedupes_and_ignores_uncited():
    """같은 URL 을 여러 번 인용해도 한 번만, 인용하지 않은 도구 결과는 넣지 않는다.

    대상: tools.extract_sources
    구분: 경계
    우선순위: 중
    사전조건: 도구 출력 URL 2개, 답변은 그중 하나를 두 번 인용
    절차: extract_sources(messages)
    기대결과: Source 1건
    """
    answer = AIMessage('A (https://docs.python.org) B (https://docs.python.org)')
    assert [s.url for s in extract_sources([tool(WEB_OUTPUT), answer])] == [
        'https://docs.python.org'
    ]


@pytest.mark.parametrize('case', ['empty', 'no_tool_output', 'answer_without_url'], ids=lambda c: c)
def test_extract_sources_empty_cases(case):
    """메시지가 없거나, 도구 출력이 없거나, 답변에 URL 이 없으면 빈 목록이다.

    대상: tools.extract_sources
    구분: 경계
    우선순위: 중
    사전조건: 각 경우의 메시지 목록
    절차: extract_sources(messages)
    기대결과: []
    """
    messages = {
        'empty': [],
        'no_tool_output': [AIMessage('사실 (https://docs.python.org)')],
        'answer_without_url': [tool(WEB_OUTPUT), AIMessage('출처 없는 요약')],
    }[case]
    assert extract_sources(messages) == []


def test_extract_sources_from_real_tool_output(tmp_path):
    """실제 도구 함수가 만든 출력 형식에서도 제목과 URL 을 읽어 낸다.

    대상: tools.extract_sources + web_search/retrieve 출력 형식
    구분: 정상
    우선순위: 상
    사전조건: 가짜 검색 결과 2건, 가짜 임베딩으로 web_search·retrieve 실행
    절차: 두 도구 출력과 URL 2개를 인용한 답변으로 extract_sources
    기대결과: 두 출처의 제목이 검색 결과의 title 과 같음
    """
    from conftest import FakeSearchClient
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from book_agent.services import VectorStoreService, WebSearchService
    from book_agent.tools import build_research_tools

    results = [
        {'title': '파이썬 공식 문서', 'url': 'https://docs.python.org', 'content': '파이썬 설치'},
        {'title': 'PEP 8', 'url': 'https://peps.python.org/pep-0008', 'content': '코딩 스타일'},
    ]
    vectors = VectorStoreService(
        persist_dir=tmp_path / 'chroma',
        embeddings=DeterministicFakeEmbedding(size=16),
        collection_name='test',
    )
    web_search, retrieve = build_research_tools(
        WebSearchService(FakeSearchClient(results), tmp_path / 'res'), vectors
    )
    outputs = [web_search.invoke({'query': 'q'}), retrieve.invoke({'query': '파이썬'})]
    answer = AIMessage(
        '- 설치 (https://docs.python.org)\n- 스타일 (https://peps.python.org/pep-0008)'
    )
    messages = [tool(o, str(i)) for i, o in enumerate(outputs)] + [answer]
    assert {s.url: s.title for s in extract_sources(messages)} == {
        'https://docs.python.org': '파이썬 공식 문서',
        'https://peps.python.org/pep-0008': 'PEP 8',
    }


# ------------------------------------------------------------------ 모델
def test_research_note_without_sources_is_compatible():
    """sources 가 없던 예전 형식의 조사 노트도 그대로 만들어진다(이전 체크포인트 호환).

    대상: models.ResearchNote.sources
    구분: 경계
    우선순위: 상
    사전조건: topic, summary 만 있는 데이터
    절차: ResearchNote.model_validate(data)
    기대결과: 오류 없이 sources=[]
    """
    assert ResearchNote.model_validate({'topic': 't', 'summary': 's'}).sources == []


def test_research_note_keeps_structured_sources():
    """출처는 제목과 URL 을 가진 Source 목록으로 저장된다.

    대상: models.ResearchNote, models.Source
    구분: 정상
    우선순위: 중
    사전조건: 출처 2건
    절차: ResearchNote 생성 후 model_dump
    기대결과: sources 가 [{'title', 'url'}, ...] 형태로 직렬화
    """
    note = ResearchNote(
        topic='t',
        summary='s',
        sources=[Source(title='A', url='https://a'), Source(title='B', url='https://b')],
    )
    assert note.model_dump()['sources'] == [
        {'title': 'A', 'url': 'https://a'},
        {'title': 'B', 'url': 'https://b'},
    ]


# ------------------------------------------------------------------ researcher 노드
def test_researcher_stores_sources_from_tool_results(make_ctx):
    """researcher 는 도구가 돌려준 문서 중 답변에 인용된 것을 출처로 저장한다.

    대상: nodes.ResearcherNode + tools.extract_sources
    구분: 정상
    우선순위: 상
    사전조건: 하위 에이전트가 [도구 출력, 최종 답변(실제 URL 1개 + 지어낸 URL 1개)] 반환
    절차: 노드 호출
    기대결과: 노트의 sources 에 실제 URL 1건만 제목과 함께 저장, summary 는 최종 답변
    """
    answer = AIMessage('- 사실 (https://docs.python.org)\n- 지어낸 사실 (https://fake.example)')
    ctx = make_ctx(researcher=agent_returning(tool(WEB_OUTPUT), answer))

    cmd = ResearcherNode()(
        {'tasks': [Task(agent='researcher', description='조사')]}, Runtime(context=ctx)
    )

    [note] = cmd.update['research_notes']
    assert [(s.title, s.url) for s in note.sources] == [
        ('파이썬 공식 문서', 'https://docs.python.org')
    ]
    assert note.summary == answer.text


# ------------------------------------------------------------------ 체크포인트 / 출력
def test_checkpoint_restores_sources(tmp_path, make_ctx):
    """조사 노트의 출처(Source)도 SQLite 체크포인트에서 타입 그대로 복원된다.

    대상: app.CHECKPOINT_TYPES (Source 등록)
    구분: 정상
    우선순위: 상
    사전조건: 출처 1건이 붙은 조사 노트를 만드는 1턴 실행 후 체크포인터 닫음
    절차: 같은 DB 를 다시 열어 상태 조회
    기대결과: research_notes[0].sources[0] 이 Source 타입, 제목·URL 유지
    """
    agent = agent_returning(tool(WEB_OUTPUT), AIMessage('- 사실 (https://docs.python.org)'))
    ctx = make_ctx(plans=[plan('researcher'), plan('communicator')], researcher=agent)
    db = tmp_path / 'cp.sqlite'
    with open_checkpointer(db) as saver:
        build_graph(checkpointer=saver).invoke(
            {'messages': [HumanMessage('책')]}, config(), context=ctx
        )

    with open_checkpointer(db) as saver:
        notes = build_graph(checkpointer=saver).get_state(config()).values['research_notes']

    [source] = notes[0].sources
    assert type(source) is Source
    assert (source.title, source.url) == ('파이썬 공식 문서', 'https://docs.python.org')


def test_run_turn_reports_source_count(make_ctx, capsys):
    """조사 완료 줄에 저장된 출처 수가 함께 표시된다.

    대상: app.run_turn
    구분: 정상
    우선순위: 하
    사전조건: researcher 가 출처 1건을 남김
    절차: run_turn 실행
    기대결과: '[researcher] 조사 완료 (1회차, 출처 1건)'
    """
    agent = agent_returning(tool(WEB_OUTPUT), AIMessage('- 사실 (https://docs.python.org)'))
    ctx = make_ctx(plans=[plan('researcher'), plan('communicator')], researcher=agent)

    run_turn(
        build_graph(checkpointer=InMemorySaver()),
        {'messages': [HumanMessage('책')]},
        config(),
        ctx,
    )

    assert '[researcher] 조사 완료 (1회차, 출처 1건)' in capsys.readouterr().out
