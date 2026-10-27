"""그래프 노드(nodes.py) 시나리오. 노드를 그래프 없이 직접 호출한다."""

import pytest
from conftest import capture_prompt, plan
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.runtime import Runtime

from book_agent.models import ResearchNote, Task
from book_agent.nodes import (
    CommunicatorNode,
    ContentStrategistNode,
    ResearcherNode,
    RoutingError,
    _format_notes,
    _format_task,
    supervisor,
)


def pending(agent, description='작업'):
    return Task(agent=agent, description=description)


# ------------------------------------------------------------------ 헬퍼
def test_format_notes_keeps_latest_five():
    """조사 노트는 최근 5개만 '## 주제' 형식으로 프롬프트에 넣는다.

    대상: nodes._format_notes
    구분: 경계
    우선순위: 중
    사전조건: 노트 7개
    절차: _format_notes(notes)
    기대결과: 3~7번만 포함, 1~2번 제외
    """
    notes = [ResearchNote(topic=f'주제{i}', summary=f'요약{i}') for i in range(1, 8)]
    text = _format_notes(notes)
    assert '## 주제1\n' not in text and '## 주제2\n' not in text
    assert all(f'## 주제{i}\n요약{i}' in text for i in range(3, 8))


def test_format_task_keeps_latest_six_with_status():
    """작업 기록은 최근 6개만 '- [상태] agent: 내용' 형식으로 넣는다.

    대상: nodes._format_task
    구분: 경계
    우선순위: 중
    사전조건: 작업 8개(첫 작업은 done)
    절차: _format_task(tasks)
    기대결과: 6줄, 첫 두 작업 제외, 형식 일치
    """
    tasks = [pending('researcher', f'작업{i}') for i in range(8)]
    tasks[2] = tasks[2].complete()
    lines = _format_task(tasks).splitlines()
    assert len(lines) == 6
    assert lines[0] == '- [done] researcher: 작업2'
    assert lines[-1] == '- [pending] researcher: 작업7'


def test_format_helpers_empty():
    """노트/작업이 없으면 안내 문구를 넣는다.

    대상: nodes._format_notes, nodes._format_task
    구분: 경계
    우선순위: 하
    사전조건: 빈 목록
    절차: 두 헬퍼 호출
    기대결과: '아직 조사 노트가 없습니다.', '아직 수행한 작업이 없습니다.'
    """
    assert _format_notes([]) == '아직 조사 노트가 없습니다.'
    assert _format_task([]) == '아직 수행한 작업이 없습니다.'


# ------------------------------------------------------------------ supervisor
@pytest.mark.parametrize('agent', ['researcher', 'content_strategist', 'communicator'])
def test_supervisor_routes_to_planned_agent(make_ctx, agent):
    """supervisor 는 플래너가 고른 agent 로 이동하고 pending 작업을 하나 추가한다.

    대상: nodes.supervisor
    구분: 정상
    우선순위: 상
    사전조건: 조사 횟수 0, 플래너가 해당 agent 선택
    절차: supervisor(state, runtime)
    기대결과: goto=agent, tasks 에 같은 agent/description 의 pending 작업 1개
    """
    ctx = make_ctx(plans=[plan(agent, '할 일')])

    cmd = supervisor({'messages': [HumanMessage('파이썬 책')]}, Runtime(context=ctx))

    assert cmd.goto == agent
    [task] = cmd.update['tasks']
    assert (task.agent, task.description, task.status) == (agent, '할 일', 'pending')


def test_supervisor_allows_research_below_limit(make_ctx):
    """조사 횟수가 한도 미만이면 조사를 허용한다.

    대상: nodes.supervisor
    구분: 경계
    우선순위: 상
    사전조건: max_research_rounds=2, research_rounds=1
    절차: 플래너가 researcher 선택
    기대결과: goto=researcher
    """
    ctx = make_ctx(plans=[plan('researcher')], max_research_rounds=2)
    cmd = supervisor({'research_rounds': 1}, Runtime(context=ctx))
    assert cmd.goto == 'researcher'


def test_supervisor_forces_writing_when_research_limit_reached(make_ctx):
    """조사 한도에 닿으면 LLM 이 조사를 골라도 content_strategist 로 넘긴다.

    대상: nodes.supervisor (무한 조사 방지)
    구분: 경계
    우선순위: 상
    사전조건: max_research_rounds=2, research_rounds=2
    절차: 플래너가 researcher('더 조사') 선택
    기대결과: goto=content_strategist, 설명에 한도(2회)와 원래 요청 포함
    """
    ctx = make_ctx(plans=[plan('researcher', '더 조사')], max_research_rounds=2)

    cmd = supervisor({'research_rounds': 2}, Runtime(context=ctx))

    assert cmd.goto == 'content_strategist'
    [task] = cmd.update['tasks']
    assert task.agent == 'content_strategist'
    assert '2회' in task.description and '더 조사' in task.description


def test_supervisor_zero_limit_disables_research(make_ctx):
    """조사 한도가 0 이면 첫 턴부터 조사 없이 목차 작성으로 간다.

    대상: nodes.supervisor
    구분: 경계
    우선순위: 중
    사전조건: max_research_rounds=0, research_rounds 없음
    절차: 플래너가 researcher 선택
    기대결과: goto=content_strategist
    """
    ctx = make_ctx(plans=[plan('researcher')], max_research_rounds=0)
    assert supervisor({}, Runtime(context=ctx)).goto == 'content_strategist'


def test_supervisor_does_not_override_other_agents_at_limit(make_ctx):
    """한도에 닿아도 조사가 아닌 선택(communicator)은 그대로 따른다.

    대상: nodes.supervisor
    구분: 경계
    우선순위: 중
    사전조건: research_rounds=한도
    절차: 플래너가 communicator 선택
    기대결과: goto=communicator
    """
    ctx = make_ctx(plans=[plan('communicator')], max_research_rounds=1)
    assert supervisor({'research_rounds': 1}, Runtime(context=ctx)).goto == 'communicator'


def test_supervisor_prompt_contains_context(make_ctx):
    """플래너가 받는 프롬프트에 조사 횟수/한도, 목차, 노트, 작업 기록, 시각, 대화가 들어간다.

    대상: nodes.supervisor + prompts.SUPERVISOR_PROMPT
    구분: 정상
    우선순위: 상
    사전조건: 목차 저장, 노트 1개, 작업 1개, 조사 1/3회, 사용자 메시지 '안녕'
    절차: supervisor 실행 후 플래너 입력 메시지 확인
    기대결과: system 메시지에 '1/3', 목차, 노트, 작업 기록, 'YYYY-' 시각 / 마지막은 사용자 메시지
    """
    seen = {}
    ctx = make_ctx(planner=capture_prompt(seen, plan('communicator')), max_research_rounds=3)
    ctx.outline_repo.save('# 기존 목차')
    state = {
        'messages': [HumanMessage('안녕')],
        'research_rounds': 1,
        'research_notes': [ResearchNote(topic='역사', summary='1991년')],
        'tasks': [pending('researcher', '역사 조사').complete()],
    }

    supervisor(state, Runtime(context=ctx))

    system, human = seen['messages']
    assert isinstance(system, SystemMessage)
    for expected in (
        '1/3',
        '# 기존 목차',
        '## 역사\n1991년',
        '- [done] researcher: 역사 조사',
        '20',
    ):
        assert expected in system.content
    assert human.content == '안녕'


def test_supervisor_propagates_planner_error(make_ctx):
    """플래너(LLM) 오류는 숨기지 않고 그대로 올려 그래프 재시도 정책이 처리하게 한다.

    대상: nodes.supervisor
    구분: 예외
    우선순위: 중
    사전조건: 플래너가 RuntimeError 발생
    절차: supervisor 실행
    기대결과: 같은 RuntimeError
    """

    def boom(_):
        raise RuntimeError('LLM down')

    ctx = make_ctx(planner=RunnableLambda(boom))
    with pytest.raises(RuntimeError, match='LLM down'):
        supervisor({}, Runtime(context=ctx))


# ------------------------------------------------------------------ 작업 노드 공통
NODES = [ResearcherNode(), ContentStrategistNode(), CommunicatorNode()]


@pytest.mark.parametrize('node', NODES, ids=lambda n: n.name)
def test_worker_rejects_mismatched_task(make_ctx, node):
    """현재 작업이 다른 agent 의 것이면 RoutingError.

    대상: nodes.WorkerNode.__call__
    구분: 예외
    우선순위: 상
    사전조건: 현재 작업의 agent 가 노드 이름과 다름
    절차: 노드 호출
    기대결과: RoutingError
    """
    other = 'communicator' if node.name != 'communicator' else 'researcher'
    with pytest.raises(RoutingError):
        node({'tasks': [pending(other)]}, Runtime(context=make_ctx()))


@pytest.mark.parametrize('node', NODES, ids=lambda n: n.name)
def test_worker_rejects_already_done_task(make_ctx, node):
    """이미 완료된 작업을 다시 받으면 RoutingError (중복 실행 방지).

    대상: nodes.WorkerNode.__call__
    구분: 예외
    우선순위: 상
    사전조건: 현재 작업이 같은 agent 의 done 작업
    절차: 노드 호출
    기대결과: RoutingError
    """
    done = pending(node.name).complete()
    with pytest.raises(RoutingError):
        node({'tasks': [done]}, Runtime(context=make_ctx()))


@pytest.mark.parametrize('node', NODES, ids=lambda n: n.name)
def test_worker_without_tasks_raises_lookup_error(make_ctx, node):
    """작업 없이 작업 노드가 실행되면 LookupError.

    대상: nodes.WorkerNode.__call__
    구분: 예외
    우선순위: 중
    사전조건: tasks 없음
    절차: 노드 호출
    기대결과: LookupError
    """
    with pytest.raises(LookupError):
        node({}, Runtime(context=make_ctx()))


# ------------------------------------------------------------------ researcher
def test_researcher_adds_note_and_increments_round(make_ctx):
    """researcher 는 조사 노트를 남기고 조사 횟수를 1 늘린 뒤 supervisor 로 돌아간다.

    대상: nodes.ResearcherNode
    구분: 정상
    우선순위: 상
    사전조건: pending researcher 작업, research_rounds=1
    절차: 노드 호출
    기대결과: goto=supervisor, 노트(topic=작업 설명, summary=에이전트 답변), rounds=2, 작업 done
    """
    task = pending('researcher', '파이썬 역사')
    ctx = make_ctx(summary='- 1991년 공개 (https://python.org)')

    cmd = ResearcherNode()({'tasks': [task], 'research_rounds': 1}, Runtime(context=ctx))

    assert cmd.goto == 'supervisor'
    [note] = cmd.update['research_notes']
    assert (note.topic, note.summary) == ('파이썬 역사', '- 1991년 공개 (https://python.org)')
    assert cmd.update['research_rounds'] == 2
    [completed] = cmd.update['tasks']
    assert completed.id == task.id and completed.status == 'done'


def test_researcher_first_round_starts_at_one(make_ctx):
    """조사 횟수 기록이 없으면 첫 조사 후 1 이 된다.

    대상: nodes.ResearcherNode
    구분: 경계
    우선순위: 중
    사전조건: research_rounds 키 없음
    절차: 노드 호출
    기대결과: research_rounds=1
    """
    cmd = ResearcherNode()({'tasks': [pending('researcher')]}, Runtime(context=make_ctx()))
    assert cmd.update['research_rounds'] == 1


def test_researcher_sends_fresh_conversation(make_ctx):
    """하위 에이전트에는 메인 대화 대신 조사 요청 메시지 하나만 전달된다.

    대상: nodes.ResearcherNode + prompts.RESEARCH_REQUEST
    구분: 정상
    우선순위: 상
    사전조건: 메인 대화 1건, 기존 노트 1건, 목차 저장됨
    절차: 노드 호출 후 하위 에이전트 입력 확인
    기대결과: HumanMessage 1개, 임무/기존 노트/목차 포함, 메인 대화 미포함
    """
    received = {}

    def fake_agent(inputs):
        received.update(inputs)
        return {'messages': [AIMessage('요약')]}

    ctx = make_ctx(researcher=RunnableLambda(fake_agent))
    ctx.outline_repo.save('# 현재 목차')
    state = {
        'tasks': [pending('researcher', '최신 버전')],
        'messages': [HumanMessage('메인 대화')],
        'research_notes': [ResearchNote(topic='기존', summary='이미 아는 사실')],
    }

    ResearcherNode()(state, Runtime(context=ctx))

    [request] = received['messages']
    assert isinstance(request, HumanMessage)
    for expected in ('최신 버전', '이미 아는 사실', '# 현재 목차'):
        assert expected in request.content
    assert '메인 대화' not in request.content


def test_researcher_uses_last_agent_message(make_ctx):
    """하위 에이전트의 여러 메시지 중 마지막 답변만 요약으로 쓴다.

    대상: nodes.ResearcherNode
    구분: 정상
    우선순위: 중
    사전조건: 에이전트가 [도구 호출 중간 메시지, 최종 답변] 반환
    절차: 노드 호출
    기대결과: summary='최종 답변'
    """
    agent = RunnableLambda(lambda _: {'messages': [AIMessage('중간'), AIMessage('최종 답변')]})
    ctx = make_ctx(researcher=agent)
    cmd = ResearcherNode()({'tasks': [pending('researcher')]}, Runtime(context=ctx))
    assert cmd.update['research_notes'][0].summary == '최종 답변'


# ------------------------------------------------------------------ content_strategist
def test_content_strategist_saves_outline_and_queues_report(make_ctx):
    """목차를 저장하고, communicator 보고 작업을 후속으로 추가한다.

    대상: nodes.ContentStrategistNode
    구분: 정상
    우선순위: 상
    사전조건: pending content_strategist 작업, LLM 답변 '# 1장 시작하기'
    절차: 노드 호출
    기대결과: outline.md 저장, goto=communicator, tasks=[완료본, pending communicator]
    """
    ctx = make_ctx(replies=['# 1장 시작하기'])
    task = pending('content_strategist')

    cmd = ContentStrategistNode()({'tasks': [task]}, Runtime(context=ctx))

    assert ctx.outline_repo.load() == '# 1장 시작하기'
    assert cmd.goto == 'communicator'
    completed, follow_up = cmd.update['tasks']
    assert completed.id == task.id and completed.status == 'done'
    assert follow_up.agent == 'communicator' and follow_up.status == 'pending'
    assert set(cmd.update) == {'tasks'}  # 메시지는 추가하지 않는다


def test_content_strategist_follow_up_ids_are_unique(make_ctx):
    """후속 보고 작업은 실행할 때마다 새 id 로 만들어진다.

    대상: nodes.ContentStrategistNode.follow_up
    구분: 경계
    우선순위: 중
    사전조건: 노드 2회 실행
    절차: 두 실행의 후속 작업 id 비교
    기대결과: 서로 다른 id (reducer 가 덮어쓰지 않음)
    """
    ctx = make_ctx(replies=['# a', '# b'])
    node = ContentStrategistNode()
    first = node({'tasks': [pending('content_strategist')]}, Runtime(context=ctx))
    second = node({'tasks': [pending('content_strategist')]}, Runtime(context=ctx))
    assert first.update['tasks'][1].id != second.update['tasks'][1].id


def test_content_strategist_prompt_contains_outline_and_notes(make_ctx):
    """목차 작성 LLM 에 작업 설명, 기존 목차, 조사 노트, 대화가 전달된다.

    대상: nodes.ContentStrategistNode + prompts.CONTENT_STRATEGIST_PROMPT
    구분: 정상
    우선순위: 상
    사전조건: 기존 목차, 노트 1개, 사용자 메시지
    절차: 노드 호출 후 LLM 입력 확인
    기대결과: system 메시지에 작업/목차/노트, 마지막 메시지는 사용자 요청
    """
    seen = {}
    ctx = make_ctx(llm=capture_prompt(seen, AIMessage('# 새 목차')))
    ctx.outline_repo.save('# 기존 목차')
    state = {
        'tasks': [pending('content_strategist', '3장 추가')],
        'research_notes': [ResearchNote(topic='역사', summary='1991년')],
        'messages': [HumanMessage('3장을 추가해 줘')],
    }

    ContentStrategistNode()(state, Runtime(context=ctx))

    system, human = seen['messages']
    for expected in ('3장 추가', '# 기존 목차', '## 역사\n1991년'):
        assert expected in system.content
    assert human.content == '3장을 추가해 줘'
    assert ctx.outline_repo.load() == '# 새 목차'


# ------------------------------------------------------------------ communicator
def test_communicator_replies_and_waits_for_human(make_ctx):
    """LLM 답변을 대화에 추가하고 human 노드로 이동한다.

    대상: nodes.CommunicatorNode
    구분: 정상
    우선순위: 상
    사전조건: pending communicator 작업
    절차: 노드 호출
    기대결과: goto=human, messages=[AI 답변], 후속 작업 없음
    """
    ctx = make_ctx(replies=['목차를 만들었어요. 어떠세요?'])

    cmd = CommunicatorNode()({'tasks': [pending('communicator')]}, Runtime(context=ctx))

    assert cmd.goto == 'human'
    [reply] = cmd.update['messages']
    assert isinstance(reply, AIMessage) and reply.content == '목차를 만들었어요. 어떠세요?'
    assert len(cmd.update['tasks']) == 1


def test_communicator_prompt_contains_outline(make_ctx):
    """보고 LLM 에 작업 설명과 현재 목차가 전달된다.

    대상: nodes.CommunicatorNode + prompts.COMMUNICATOR_PROMPT
    구분: 정상
    우선순위: 중
    사전조건: 목차 저장됨
    절차: 노드 호출 후 LLM 입력 확인
    기대결과: system 메시지에 작업 설명과 목차
    """
    seen = {}
    ctx = make_ctx(llm=capture_prompt(seen, AIMessage('보고')))
    ctx.outline_repo.save('# 1장')
    CommunicatorNode()({'tasks': [pending('communicator', '변경점 보고')]}, Runtime(context=ctx))
    system = seen['messages'][0]
    assert '변경점 보고' in system.content and '# 1장' in system.content


# ------------------------------------------------------------------ 대화 기록 자르기
def test_history_is_trimmed_to_token_limit(make_ctx):
    """오래된 대화는 토큰 한도에 맞춰 잘리고, 최근 메시지가 사람 메시지부터 남는다.

    대상: nodes._history (trim_messages)
    구분: 경계
    우선순위: 상
    사전조건: 메시지 40개, max_history_tokens=60
    절차: supervisor 실행 후 플래너 입력 확인
    기대결과: 일부만 남음, 마지막 메시지 유지, 첫 메시지는 HumanMessage
    """
    seen = {}
    ctx = make_ctx(planner=capture_prompt(seen, plan('communicator')), max_history_tokens=60)
    history = []
    for i in range(20):
        history += [HumanMessage(f'질문 {i} ' * 5), AIMessage(f'답변 {i} ' * 5)]

    supervisor({'messages': history}, Runtime(context=ctx))

    kept = seen['messages'][1:]  # 첫 메시지는 system
    assert 0 < len(kept) < len(history)
    assert kept[-1].content == history[-1].content
    assert isinstance(kept[0], HumanMessage)  # start_on='human'


def test_history_within_limit_is_kept_whole(make_ctx):
    """한도 안의 짧은 대화는 그대로 전달된다.

    대상: nodes._history
    구분: 정상
    우선순위: 중
    사전조건: 메시지 3개, 기본 한도 8000
    절차: supervisor 실행
    기대결과: 3개 모두 전달
    """
    seen = {}
    ctx = make_ctx(planner=capture_prompt(seen, plan('communicator')))
    history = [HumanMessage('a'), AIMessage('b'), HumanMessage('c')]
    supervisor({'messages': history}, Runtime(context=ctx))
    assert [m.content for m in seen['messages'][1:]] == ['a', 'b', 'c']
