"""
그래프 상태 (State)와 reducer 정의 모듈.

노드는 상태를 직접 수정하지 않고 '바뀐 부분'만 반환한다.
각 필드에 붙은 reducer가 이전 값과 반환 값을 합친다.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import ResearchNote, Task


def merge_tasks(left: list[Task], right: list[Task]) -> list[Task]:
    """
    작업 목록 reducer. 같은 id는 교체(완료처리)하고, 새 id는 뒤에 추가한다.

    dict는 삽입 순서를 유지하므로 교체된 작업은 원래 위치에 남는다.
    """

    merged = {task.id: task for task in left}
    merged.update({task.id: task for task in right})
    return list(merged.values())


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tasks: Annotated[list[Task], merge_tasks]
    research_notes: Annotated[list[ResearchNote], operator.add]
    research_rounds: int  # reducer 없음 -> 마지막 값으로 덮어씀


def current_task(state: State) -> Task:
    tasks = state.get('tasks', [])
    if not tasks:
        raise LookupError('진행중인 작업이 없습니다.')
    return tasks[-1]
