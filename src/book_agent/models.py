"""
도메인 모델 모듈.

- TaskPlan      : supervisor LLM 이 출력하는 구조화 스키마(LLM 전용)
- Task          : 그래프 안에서 관리하는 작업 레코드 (내부 전용)
- ResearchNote  : research가 남기는 조사 요약
"""

from datetime import datetime
from typing import Literal, Self, get_args
from uuid import uuid4

from pydantic import BaseModel, Field

Worker = Literal['researcher', 'content_strategist', 'communicator']
WORKERS: tuple[str, ...] = get_args(Worker)


class TaskPlan(BaseModel):
    """
    supervisor 의 구조화 출력 스키마.

    OpenAI strict JSON schema 모드는 기본값이 있는 필드를 허용하지 않으므로
    LLM이 채워야 하는 필드만 두고, 기본값은 하나도 두지 않는다.
    """

    agent: Worker = Field(
        description=(
            '작업을 수행할 agent. '
            'researcher : 목차 작성에 필요한 자료를 웹/벡터 DB에서 조사한다. '
            'content_strategist: 요구 사항과 조사 자료를 바탕으로 목차를 작성/수정한다. '
            'communicator: 사용자에게 진행 상황을 보고하거나 추가 정보를 묻는다.'
        )
    )
    description: str = Field(description='담당 agent가 수행할 구체적인 작업 내용')


class Task(TaskPlan):
    """
    그래프 상태에 누적되는 작업 레코드.

    TaskPlan을 상속하고 내부 관리용 필드(id, 상태, 시각)를 추가한다.
    상태를 직접 바꾸지 않고 complete()로 '완료된 사본'을 만든다 (불변 객체 스타일.)
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    status: Literal['pending', 'done'] = 'pending'
    created_at: datetime = Field(default_factory=datetime.now)
    done_at: datetime | None = None

    def complete(self) -> Self:
        return self.model_copy(update={'status': 'done', 'done_at': datetime.now()})


class ResearchNote(BaseModel):
    """researcher 가 한 번의 조사로 남기는 요약(출처 포함.)"""

    topic: str
    summary: str
    created_at: datetime = Field(default_factory=datetime.now)
