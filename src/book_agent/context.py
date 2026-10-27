"""
런타임 컨텍스트(의존성 묶음) 모듈.

그래프 상태 (State)는 '대화가 진행되며 바뀌는 데이터'이고,
AppContext 는 '실행 내내 바뀌지 않는 의존성(모델, 저장소, 하위 에이전트)'이다.
노드는 runtime.context로 이 객체를 받는다(LangGraph Runtime 주입).
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from .models import TaskPlan
from .repositories import OutlineRepository


@dataclass(frozen=True, slots=True)
class AppContext:
    llm: BaseChatModel
    planner: Runnable[dict[str, Any], TaskPlan]
    researcher: Runnable[dict[str, Any], dict[str, Any]]
    outline_repo: OutlineRepository
    max_research_rounds: int = 2
    max_history_tokens: int = 8000
