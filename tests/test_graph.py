"""그래프(graph.py) 통합 시나리오. 가짜 모델로 전체 흐름을 돌린다."""

import pytest
from conftest import plan
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from book_agent.graph import LLM_RETRY, build_graph


def config(thread='t1', limit=40):
    return {'configurable': {'thread_id': thread}, 'recursion_limit': limit}


def start(text='파이썬 책'):
    return {'messages': [HumanMessage(text)]}


def test_graph_registers_all_nodes():
    """그래프에 다섯 노드가 모두 등록된다.

    대상: graph.build_graph
    구분: 정상
    우선순위: 상
    사전조건: 체크포인터 없음
    절차: build_graph()
    기대결과: supervisor, researcher, content_strategist, communicator, human 포함
    """
    nodes = set(build_graph().nodes)
    assert {'supervisor', 'researcher', 'content_strategist', 'communicator', 'human'} <= nodes


def test_graph_edges_follow_declared_destinations():
    """Command 반환 타입과 destinations 선언대로 이동 경로가 그려진다.

    대상: graph.build_graph (destinations)
    구분: 정상
    우선순위: 중
    사전조건: 없음
    절차: get_graph().edges 의 (출발, 도착) 확인
    기대결과: START→supervisor, supervisor→3 작업자, researcher→supervisor,
        content_strategist→communicator, communicator→human, human→supervisor
    """
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    expected = {
        ('__start__', 'supervisor'),
        ('supervisor', 'researcher'),
        ('supervisor', 'content_strategist'),
        ('supervisor', 'communicator'),
        ('researcher', 'supervisor'),
        ('content_strategist', 'communicator'),
        ('communicator', 'human'),
        ('human', 'supervisor'),
    }
    assert expected <= edges


def test_retry_policy_settings():
    """LLM 노드 재시도 정책은 3회, 1초부터 2배 백오프, 지터 사용이다.

    대상: graph.LLM_RETRY
    구분: 정상
    우선순위: 하
    사전조건: 없음
    절차: 정책 속성 확인
    기대결과: max_attempts=3, initial_interval=1.0, backoff_factor=2.0, jitter=True
    """
    assert LLM_RETRY.max_attempts == 3
    assert (LLM_RETRY.initial_interval, LLM_RETRY.backoff_factor) == (1.0, 2.0)
    assert LLM_RETRY.jitter is True


def test_full_turn_pauses_at_human(make_ctx):
    """조사 → 목차 작성 → 보고 순서로 진행하고 human 노드에서 멈춘다.

    대상: graph 전체 흐름
    구분: 정상
    우선순위: 상
    사전조건: 플래너 [researcher, content_strategist], LLM 답변 [목차, 보고]
    절차: 사용자 메시지로 invoke
    기대결과: next=('human',), 작업 3개 모두 done, 조사 1회, 노트 1개, 마지막 메시지는 보고,
        목차 저장
    """
    ctx = make_ctx(
        plans=[plan('researcher', '조사'), plan('content_strategist', '목차')],
        replies=['# 1장', '목차를 만들었습니다.'],
    )
    graph = build_graph(checkpointer=InMemorySaver())

    graph.invoke(start(), config(), context=ctx)

    state = graph.get_state(config())
    assert state.next == ('human',)
    assert [(t.agent, t.status) for t in state.values['tasks']] == [
        ('researcher', 'done'),
        ('content_strategist', 'done'),
        ('communicator', 'done'),
    ]
    assert state.values['research_rounds'] == 1
    assert len(state.values['research_notes']) == 1
    assert state.values['messages'][-1].content == '목차를 만들었습니다.'
    assert ctx.outline_repo.load() == '# 1장'


def test_interrupt_payload_asks_for_next_instruction(make_ctx):
    """human 노드에서 멈출 때 사용자에게 보여줄 질문이 interrupt 값으로 남는다.

    대상: nodes.human (interrupt)
    구분: 정상
    우선순위: 중
    사전조건: communicator 까지 진행
    절차: get_state().interrupts 확인
    기대결과: {'question': '다음 지시를 입력하세요.'}
    """
    ctx = make_ctx(plans=[plan('communicator')])
    graph = build_graph(checkpointer=InMemorySaver())
    graph.invoke(start(), config(), context=ctx)
    [pending_interrupt] = graph.get_state(config()).interrupts
    assert pending_interrupt.value == {'question': '다음 지시를 입력하세요.'}


def test_resume_adds_user_message_and_resets_rounds(make_ctx):
    """재개하면 사용자 입력이 대화에 추가되고 조사 횟수가 0 으로 초기화된다.

    대상: nodes.human + graph 재개
    구분: 정상
    우선순위: 상
    사전조건: 1턴에서 조사 1회 후 human 에서 대기
    절차: Command(resume='초보자용') 으로 재개
    기대결과: 메시지 순서 [책, 질문, 초보자용, 답변], research_rounds=0, 다시 human 에서 대기
    """
    ctx = make_ctx(
        plans=[plan('researcher'), plan('communicator', '질문'), plan('communicator', '답변')],
        replies=['무엇을 쓸까요?', '알겠습니다.'],
    )
    graph = build_graph(checkpointer=InMemorySaver())
    graph.invoke(start('책'), config(), context=ctx)
    assert graph.get_state(config()).values['research_rounds'] == 1

    graph.invoke(Command(resume='초보자용'), config(), context=ctx)

    state = graph.get_state(config())
    contents = [m.content for m in state.values['messages']]
    assert contents == ['책', '무엇을 쓸까요?', '초보자용', '알겠습니다.']
    assert state.values['research_rounds'] == 0
    assert state.next == ('human',)


def test_research_limit_stops_loop(make_ctx):
    """LLM 이 계속 조사만 원해도 한도 이후에는 목차 작성으로 넘어간다.

    대상: graph + nodes.supervisor (무한 조사 방지)
    구분: 경계
    우선순위: 상
    사전조건: 플래너가 항상 researcher, max_research_rounds=2
    절차: 1턴 실행
    기대결과: 작업 순서 [researcher, researcher, content_strategist, communicator], 노트 2개
    """
    ctx = make_ctx(
        plans=[plan('researcher')] * 3, replies=['# 목차', '보고'], max_research_rounds=2
    )
    graph = build_graph(checkpointer=InMemorySaver())

    graph.invoke(start(), config(), context=ctx)

    values = graph.get_state(config()).values
    agents = [t.agent for t in values['tasks']]
    assert agents == ['researcher', 'researcher', 'content_strategist', 'communicator']
    assert len(values['research_notes']) == 2


def test_research_notes_accumulate_across_turns(make_ctx):
    """조사 노트는 턴이 바뀌어도 누적된다(operator.add reducer).

    대상: state.research_notes
    구분: 정상
    우선순위: 중
    사전조건: 1턴과 2턴에서 각각 조사 1회
    절차: 1턴 실행 → resume 으로 2턴 실행
    기대결과: 노트 2개, 조사 횟수는 2턴 기준 1
    """
    ctx = make_ctx(
        plans=[plan('researcher'), plan('communicator'), plan('researcher'), plan('communicator')],
        replies=['보고1', '보고2'],
    )
    graph = build_graph(checkpointer=InMemorySaver())
    graph.invoke(start(), config(), context=ctx)
    graph.invoke(Command(resume='더 알려줘'), config(), context=ctx)

    values = graph.get_state(config()).values
    assert len(values['research_notes']) == 2
    assert values['research_rounds'] == 1


def test_threads_are_isolated(make_ctx):
    """thread_id 가 다르면 대화 상태가 섞이지 않는다.

    대상: graph + checkpointer(thread_id)
    구분: 정상
    우선순위: 상
    사전조건: 같은 체크포인터, thread A/B
    절차: A, B 에 각각 다른 메시지로 1턴 실행
    기대결과: 각 thread 의 첫 메시지가 자기 입력
    """
    ctx = make_ctx(plans=[plan('communicator'), plan('communicator')], replies=['a', 'b'])
    graph = build_graph(checkpointer=InMemorySaver())
    graph.invoke(start('A 책'), config('A'), context=ctx)
    graph.invoke(start('B 책'), config('B'), context=ctx)

    assert graph.get_state(config('A')).values['messages'][0].content == 'A 책'
    assert graph.get_state(config('B')).values['messages'][0].content == 'B 책'
    assert len(graph.get_state(config('A')).values['messages']) == 2


def test_recursion_limit_stops_runaway_graph(make_ctx):
    """조사 한도가 매우 커도 recursion_limit 에서 실행이 멈춘다.

    대상: graph (recursion_limit)
    구분: 예외
    우선순위: 중
    사전조건: 플래너가 항상 researcher, max_research_rounds=100, recursion_limit=5
    절차: 1턴 실행
    기대결과: GraphRecursionError
    """
    ctx = make_ctx(plans=[plan('researcher')] * 100, max_research_rounds=100)
    graph = build_graph(checkpointer=InMemorySaver())
    with pytest.raises(GraphRecursionError):
        graph.invoke(start(), config(limit=5), context=ctx)


class TransientAPIError(Exception):
    """일시적인 API 오류 대역(기본 retry_on 이 재시도하는 일반 예외)."""


def flaky_researcher(failures, error):
    calls = {'n': 0}

    def run(_):
        calls['n'] += 1
        if calls['n'] <= failures:
            raise error
        return {'messages': [AIMessage('요약')]}

    return RunnableLambda(run), calls


def test_transient_error_is_retried(make_ctx, fast_retry):
    """일시적 API 오류는 재시도 정책에 따라 다시 실행되어 성공한다.

    대상: graph.LLM_RETRY (researcher 노드)
    구분: 예외
    우선순위: 상
    사전조건: researcher 가 처음 2회 TransientAPIError, 3회째 성공 / 대기 시간 0
    절차: 1턴 실행
    기대결과: 3회 호출 후 정상 진행, human 에서 대기
    """
    researcher, calls = flaky_researcher(2, TransientAPIError('503'))
    ctx = make_ctx(plans=[plan('researcher'), plan('communicator')], researcher=researcher)
    graph = build_graph(checkpointer=InMemorySaver())

    graph.invoke(start(), config(), context=ctx)

    assert calls['n'] == 3
    assert graph.get_state(config()).next == ('human',)


def test_retry_gives_up_after_max_attempts(make_ctx, fast_retry):
    """재시도 한도(3회)를 넘기면 오류를 그대로 올린다.

    대상: graph.LLM_RETRY
    구분: 예외
    우선순위: 중
    사전조건: researcher 가 항상 TransientAPIError
    절차: 1턴 실행
    기대결과: 3회 호출 후 TransientAPIError
    """
    researcher, calls = flaky_researcher(99, TransientAPIError('503'))
    ctx = make_ctx(plans=[plan('researcher')], researcher=researcher)
    with pytest.raises(TransientAPIError):
        build_graph(checkpointer=InMemorySaver()).invoke(start(), config(), context=ctx)
    assert calls['n'] == 3


def test_code_bug_is_not_retried(make_ctx, fast_retry):
    """코드 버그(ValueError)는 재시도하지 않고 바로 실패한다.

    대상: graph.LLM_RETRY (기본 retry_on)
    구분: 예외
    우선순위: 중
    사전조건: researcher 가 ValueError
    절차: 1턴 실행
    기대결과: 1회 호출 후 ValueError
    """
    researcher, calls = flaky_researcher(99, ValueError('bug'))
    ctx = make_ctx(plans=[plan('researcher')], researcher=researcher)
    with pytest.raises(ValueError, match='bug'):
        build_graph(checkpointer=InMemorySaver()).invoke(start(), config(), context=ctx)
    assert calls['n'] == 1


def test_config_error_is_not_retried(make_ctx, fast_retry):
    """키 오류처럼 다시 해도 실패할 오류는 재시도하지 않는다(비용·대기 시간 절약).

    대상: graph.LLM_RETRY (errors.retry_on)
    구분: 예외
    우선순위: 상
    사전조건: researcher 가 openai.AuthenticationError(401)
    절차: 1턴 실행
    기대결과: 1회 호출 후 AuthenticationError
    """
    import httpx
    import openai

    response = httpx.Response(401, request=httpx.Request('POST', 'https://api.openai.com'))
    error = openai.AuthenticationError('bad key', response=response, body=None)
    researcher, calls = flaky_researcher(99, error)
    ctx = make_ctx(plans=[plan('researcher')], researcher=researcher)
    with pytest.raises(openai.AuthenticationError):
        build_graph(checkpointer=InMemorySaver()).invoke(start(), config(), context=ctx)
    assert calls['n'] == 1


def test_retry_policy_uses_error_classification():
    """그래프 재시도 정책은 errors.retry_on 판정을 쓴다.

    대상: graph.LLM_RETRY.retry_on
    구분: 정상
    우선순위: 중
    사전조건: 없음
    절차: 정책의 retry_on 확인
    기대결과: errors.retry_on 과 같은 함수
    """
    from book_agent.errors import retry_on

    assert LLM_RETRY.retry_on is retry_on
