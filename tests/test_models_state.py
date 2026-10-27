"""도메인 모델(models.py)과 상태 reducer(state.py) 시나리오."""

import pytest
from pydantic import ValidationError

from book_agent.models import WORKERS, ResearchNote, Task, TaskPlan
from book_agent.state import current_task, merge_tasks


def test_workers_match_literal():
    """작업자 목록은 그래프 노드 이름과 같은 세 가지다.

    대상: models.WORKERS
    구분: 정상
    우선순위: 상
    사전조건: 없음
    절차: WORKERS 값 확인
    기대결과: ('researcher', 'content_strategist', 'communicator')
    """
    assert WORKERS == ('researcher', 'content_strategist', 'communicator')


@pytest.mark.parametrize('agent', ['writer', 'research', 'supervisor', ''])
def test_task_plan_rejects_unknown_agent(agent):
    """LLM 이 목록에 없는 agent 를 내놓으면 검증 단계에서 막는다.

    대상: models.TaskPlan
    구분: 예외
    우선순위: 상
    사전조건: 없음
    절차: 허용되지 않은 agent 이름으로 TaskPlan 생성
    기대결과: ValidationError
    """
    with pytest.raises(ValidationError):
        TaskPlan(agent=agent, description='x')


def test_task_plan_requires_description():
    """description 이 빠진 계획은 거부된다.

    대상: models.TaskPlan
    구분: 예외
    우선순위: 중
    사전조건: 없음
    절차: agent 만 넣어 TaskPlan 생성
    기대결과: description 을 언급하는 ValidationError
    """
    with pytest.raises(ValidationError, match='description'):
        TaskPlan(agent='researcher')


def test_task_plan_schema_has_no_defaults():
    """OpenAI strict 모드를 위해 TaskPlan 의 모든 필드는 필수이고 enum 이 노출된다.

    대상: models.TaskPlan JSON 스키마
    구분: 정상
    우선순위: 상
    사전조건: 없음
    절차: model_json_schema() 확인
    기대결과: required={agent, description}, 기본값 없음, agent enum 은 WORKERS
    """
    schema = TaskPlan.model_json_schema()
    assert set(schema['required']) == {'agent', 'description'}
    assert all('default' not in p for p in schema['properties'].values())
    assert tuple(schema['properties']['agent']['enum']) == WORKERS


def test_task_defaults_are_pending_with_unique_id():
    """새 작업은 pending 상태이고 작업마다 id 가 다르다.

    대상: models.Task
    구분: 정상
    우선순위: 상
    사전조건: 없음
    절차: 같은 내용으로 Task 두 개 생성
    기대결과: status=pending, done_at=None, created_at 기록, 서로 다른 32자리 id
    """
    a = Task(agent='researcher', description='조사')
    b = Task(agent='researcher', description='조사')
    assert a.status == 'pending' and a.done_at is None and a.created_at is not None
    assert a.id != b.id and len(a.id) == 32


def test_task_from_plan_keeps_fields():
    """supervisor 처럼 TaskPlan 을 펼쳐 Task 를 만들면 내용이 그대로 옮겨진다.

    대상: models.Task(**TaskPlan.model_dump())
    구분: 정상
    우선순위: 중
    사전조건: TaskPlan(content_strategist, '목차 작성')
    절차: Task(**plan.model_dump())
    기대결과: agent/description 동일, Task 는 TaskPlan 의 하위 타입
    """
    p = TaskPlan(agent='content_strategist', description='목차 작성')
    task = Task(**p.model_dump())
    assert (task.agent, task.description) == (p.agent, p.description)
    assert isinstance(task, TaskPlan)


def test_task_rejects_unknown_status():
    """상태는 pending/done 만 허용된다.

    대상: models.Task.status
    구분: 예외
    우선순위: 하
    사전조건: 없음
    절차: status='running' 으로 Task 생성
    기대결과: ValidationError
    """
    with pytest.raises(ValidationError):
        Task(agent='researcher', description='x', status='running')


def test_task_complete_returns_copy():
    """complete() 는 원본을 바꾸지 않고 완료된 사본을 만든다(같은 id 유지).

    대상: models.Task.complete
    구분: 정상
    우선순위: 상
    사전조건: pending 작업 하나
    절차: task.complete() 호출
    기대결과: 원본은 pending, 사본은 done + done_at 기록 + 같은 id/created_at
    """
    task = Task(agent='communicator', description='보고')
    done = task.complete()
    assert task.status == 'pending' and task.done_at is None
    assert done.status == 'done' and done.done_at is not None
    assert (done.id, done.created_at) == (task.id, task.created_at)


def test_merge_tasks_replaces_same_id_in_place_and_appends_new():
    """같은 id 는 원래 위치에서 교체되고, 새 id 는 뒤에 붙는다.

    대상: state.merge_tasks
    구분: 정상
    우선순위: 상
    사전조건: 기존 작업 [1, 2]
    절차: [1 완료본, 3] 을 병합
    기대결과: 순서 [1, 2, 3], 상태 [done, pending, pending]
    """
    first = Task(agent='researcher', description='1')
    second = Task(agent='content_strategist', description='2')
    third = Task(agent='communicator', description='3')

    merged = merge_tasks([first, second], [first.complete(), third])

    assert [t.id for t in merged] == [first.id, second.id, third.id]
    assert [t.status for t in merged] == ['done', 'pending', 'pending']


@pytest.mark.parametrize('side', ['left', 'right'])
def test_merge_tasks_with_empty_side(side):
    """한쪽이 비어 있으면 다른 쪽 목록이 그대로 결과가 된다.

    대상: state.merge_tasks
    구분: 경계
    우선순위: 중
    사전조건: 작업 2개 목록과 빈 목록
    절차: 빈 목록을 left 또는 right 로 병합
    기대결과: 원래 작업 2개가 순서대로 남는다
    """
    tasks = [Task(agent='researcher', description='1'), Task(agent='communicator', description='2')]
    merged = merge_tasks([], tasks) if side == 'left' else merge_tasks(tasks, [])
    assert merged == tasks


def test_current_task_returns_last():
    """현재 작업은 목록의 마지막 작업이다.

    대상: state.current_task
    구분: 정상
    우선순위: 중
    사전조건: 작업 2개
    절차: current_task(state)
    기대결과: 마지막 작업 객체
    """
    tasks = [Task(agent='researcher', description='1'), Task(agent='communicator', description='2')]
    assert current_task({'tasks': tasks}) is tasks[-1]


@pytest.mark.parametrize('state', [{}, {'tasks': []}], ids=['no_key', 'empty_list'])
def test_current_task_without_tasks_raises(state):
    """작업이 없는 상태에서 작업 노드가 실행되면 LookupError.

    대상: state.current_task
    구분: 예외
    우선순위: 중
    사전조건: tasks 키 없음 / 빈 목록
    절차: current_task(state)
    기대결과: LookupError
    """
    with pytest.raises(LookupError):
        current_task(state)


def test_research_note_has_timestamp():
    """조사 노트는 생성 시각을 자동으로 기록한다.

    대상: models.ResearchNote
    구분: 정상
    우선순위: 하
    사전조건: 없음
    절차: ResearchNote(topic, summary) 생성
    기대결과: created_at 자동 기록
    """
    note = ResearchNote(topic='t', summary='s')
    assert note.created_at is not None
