"""
그래프 조립 모듈.

노드 사이의 이동은 각 노드가 반환하는  Command(goto=...)가 결정한다.
여기서는 노드 등록, 재시도 정책, 이동 가능 목적지(destinations)만 선언한다.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from .context import AppContext
from .errors import retry_on
from .models import WORKERS
from .nodes import CommunicatorNode, ContentStrategistNode, ResearcherNode, human, supervisor
from .state import State

# 일시적인 네트워크/API 오류만 재시도한다. 키/모델 오류처럼 다시 해도 실패할 오류와
# 코드 버그(ValueError 등)는 재시도하지 않는다(errors.retry_on).
LLM_RETRY = RetryPolicy(
    max_attempts=3, initial_interval=1.0, backoff_factor=2.0, jitter=True, retry_on=retry_on
)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(State, context_schema=AppContext)

    builder.add_node('supervisor', supervisor, retry_policy=LLM_RETRY)
    # WorkerNode.__call__의 반환 타입이 일반 Command이므로 목적지를 직접 알려 준다.
    # (그래프 그림의 엣지와 검증에 사용된다.)
    builder.add_node(
        'researcher', ResearcherNode(), retry_policy=LLM_RETRY, destinations=('supervisor',)
    )

    builder.add_node(
        'content_strategist',
        ContentStrategistNode(),
        retry_policy=LLM_RETRY,
        destinations=('communicator',),
    )
    builder.add_node(
        'communicator', CommunicatorNode(), retry_policy=LLM_RETRY, destinations=('human',)
    )

    builder.add_node('human', human)
    builder.add_edge(START, 'supervisor')

    graph = builder.compile(checkpointer=checkpointer)
    assert set(WORKERS) <= set(graph.nodes), 'Worker 타입과 노드 이름이 일치해야 합니다.'
    return graph
