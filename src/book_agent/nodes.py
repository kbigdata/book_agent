"""
그래프 노드 모듈.

- supervisor: 다음 작업(task)을 정하고 Command(goto=...)의 해당 노드로 이동한다.
- WorkerNode: 작업 노드의 공통 골격(템플릿 메소드).
              작업 검증 -> run() -> 완료처리 -> 후속 작업 -> 이동 을 한곳에서 처리한다.
- human : interrupt로 그래프를 멈추고 사용자 입력을 기다린다.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Literal, override

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from .context import AppContext
from .models import ResearchNote, Task, TaskPlan, Worker
from .prompts import (
    COMMUNICATOR_PROMPT,
    CONTENT_STRATEGIST_PROMPT,
    RESEARCH_REQUEST,
    SUPERVISOR_PROMPT,
)
from .state import State, current_task


class RoutingError(RuntimeError):
    """현재 작업과 맞지 않는 노드가 실행되었을 때 발생한다."""


# --------------------------------------------------- 공통 헬퍼
def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _history(state: State, ctx: AppContext) -> list[AnyMessage]:
    """최근 대화만 토큰 한도 안에서 남긴다.(오래된 메시지부터 잘라냄)."""
    return trim_messages(
        state.get('messages', []),
        max_tokens=ctx.max_history_tokens,
        token_counter=count_tokens_approximately,
        strategy='last',
        start_on='human',
    )


def _format_notes(notes: list[ResearchNote], limit: int = 5) -> str:
    if not notes:
        return '아직 조사 노트가 없습니다.'
    return '\n\n'.join(f'## {n.topic}\n{n.summary}' for n in notes[-limit:])


def _format_task(tasks: list[Task], limit: int = 6) -> str:
    if not tasks:
        return '아직 수행한 작업이 없습니다.'
    return '\n'.join(f'- [{t.status}] {t.agent}: {t.description}' for t in tasks[-limit:])


def supervisor(state: State, runtime: Runtime[AppContext]) -> Command[Worker]:
    ctx = runtime.context
    rounds = state.get('research_rounds', 0)

    chain = SUPERVISOR_PROMPT | ctx.planner
    plan: TaskPlan = chain.invoke(
        {
            'now': _now(),
            'outline': ctx.outline_repo.load(),
            'research_notes': _format_notes(state.get('research_notes', [])),
            'task_log': _format_task(state.get('tasks', [])),
            'research_rounds': rounds,
            'max_research_rounds': ctx.max_research_rounds,
            'messages': _history(state, ctx),
        }
    )

    # 무한 조사 루프 방지: 한도에 닿으면 LLM 판단과 관계없이 목차 작성으로 넘어간다.
    if plan.agent == 'researcher' and rounds >= ctx.max_research_rounds:
        plan = TaskPlan(
            agent='content_strategist',
            description=f'조사 한도 ({ctx.max_research_rounds}회)에 도달했다.'
            f'지금까지의 자료로 목차를 작성한다. 원래 요청 : {plan.description}',
        )
    return Command(update={'tasks': [Task(**plan.model_dump())]}, goto=plan.agent)


# ------------------------------------------------------------------------ 작업노드
class WorkerNode(ABC):
    """
    작업 노드의 기반 클래스(템플릿 메서드 패턴)

    하위 클래스는 name, goto 를 정하고 run() 만 구현한다.
    작업 검증, 완료 처리, 후속 작업 생성, 이동은 __call__ 이 공통으로 처리한다.
    """

    name: ClassVar[Worker]
    goto: ClassVar[str]
    follow_up: ClassVar[TaskPlan | None] = None

    def __call__(self, state: State, runtime: Runtime[AppContext]) -> Command:
        task = current_task(state)
        if task.agent != self.name or task.status != 'pending':
            raise RoutingError(f'{self.name} 노드가 다른 작업을 받았습니다: {task}')

        update = self.run(state, runtime.context, task)

        tasks = [task.complete()]

        if self.follow_up is not None:
            tasks.append(Task(**self.follow_up.model_dump()))
        return Command(update={**update, 'tasks': tasks}, goto=self.goto)

    @abstractmethod
    def run(self, state: State, ctx: AppContext, task: Task) -> dict[str, Any]:
        """노드 고유의 일을 하고, 상태에 반영할 변경분(dict)을 반환한다."""


class ResearcherNode(WorkerNode):
    name = 'researcher'
    goto = 'supervisor'

    @override
    def run(self, state: State, ctx: AppContext, task: Task) -> dict[str, Any]:
        request = RESEARCH_REQUEST.format(
            mission=task.description,
            now=_now(),
            outline=ctx.outline_repo.load(),
            known=_format_notes(state.get('research_notes', [])),
        )
        # 하위 에이전트는 '새 대화'로 실행한다. 도구 호출 기록이 메인 대화를 오염시키지 않는다.
        result = ctx.researcher.invoke({'messages': [HumanMessage(request)]})
        summary = result['messages'][-1].text

        return {
            'research_notes': [ResearchNote(topic=task.description, summary=summary)],
            'research_rounds': state.get('research_rounds', 0) + 1,
        }


class ContentStrategistNode(WorkerNode):
    name = 'content_strategist'
    goto = 'communicator'
    follow_up = TaskPlan(
        agent='communicator',
        description='갱신된 목차의 핵심 변경점을 사용자에게 보고하고 피드백을 요청한다.',
    )

    @override
    def run(self, state: State, ctx: AppContext, task: Task) -> dict[str, Any]:
        chain = CONTENT_STRATEGIST_PROMPT | ctx.llm | StrOutputParser()
        outline = chain.invoke(
            {
                'mission': task.description,
                'outline': ctx.outline_repo.load(),
                'research_notes': _format_notes(state.get('research_notes', [])),
                'messages': _history(state, ctx),
            }
        )
        ctx.outline_repo.save(outline)
        return {}


class CommunicatorNode(WorkerNode):
    name = 'communicator'
    goto = 'human'

    @override
    def run(self, state: State, ctx: AppContext, task: Task) -> dict[str, Any]:
        reply = (COMMUNICATOR_PROMPT | ctx.llm).invoke(
            {
                'mission': task.description,
                'outline': ctx.outline_repo.load(),
                'messages': _history(state, ctx),
            }
        )

        return {'messages': [reply]}


# ------------------------------------------------------------- human
def human(state: State) -> Command[Literal['supervisor']]:
    """
    그래프를 일시 정지하고 사용자 입력을 기다린다.

    재개되면 이 노드는 처음부터 다시 실행되므로 interrupt() 앞에는 부수효과를 두지 않는다.
    """
    reply: str = interrupt({'question': '다음 지시를 입력하세요.'})
    return Command(
        update={'messages': [HumanMessage(reply)], 'research_rounds': 0}, goto='supervisor'
    )
