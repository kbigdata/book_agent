"""
researcher  에이전트가 사용하는 도구 모듈.

도구는 서비스 객체를 클로저로 캡처한다.
전역 변수 없이도 LLM에는 'query' 인자 하나만 노출된다.
"""

import re

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from .models import Source
from .services import VectorStoreService, WebSearchService, format_documents

# 도구 출력의 문서 목록 줄: web_search 는 '- 제목 (URL)', retrieve 는 '[번호] 제목 (URL)'.
_LISTED = re.compile(
    r'^\s*(?:- |\[\d+\] )(?P<title>.*) \((?P<url>https?://[^\s()]+)\)\s*$', re.MULTILINE
)
_URL = re.compile(r'https?://[^\s()<>\[\]"\']+')


def build_research_tools(
    web: WebSearchService, vectors: VectorStoreService, top_k: int = 5
) -> list[BaseTool]:

    @tool
    def web_search(query: str) -> str:
        """웹에서 최신 정보를 검색해 벡터DB에 저장하고, 찾은 문서의 제목과 URL 목록을 반환한다.
        벡터 DB에 원하는 자료가 없을 때 사용한다. 복합적인 질문은 나눠서 여러 번 호출한다."""

        docs = web.search(query)
        added = vectors.add_documents(docs)
        listing = '\n'.join(f'- {d.metadata["title"]} ({d.metadata["source"]})' for d in docs)
        return f'{len(docs)}건 검색, 새 청크 {added}개 저장\n {listing}'

    @tool
    def retrieve(query: str) -> str:
        """벡터 DB에서 query 와 관련된 자료를 찾아 [번호] 제목(출처)과 본문 발췌를 반환한다."""
        return format_documents(vectors.search(query, k=top_k))

    return [web_search, retrieve]


def extract_sources(messages: list[AnyMessage]) -> list[Source]:
    """
    researcher 의 최종 답변에 인용된 URL 중 검색 도구가 실제로 돌려준 URL 만 출처로 남긴다.

    LLM 이 지어낸 URL 은 도구 출력에 없으므로 걸러진다. 제목은 도구 출력에서 가져오고,
    순서는 답변에 처음 인용된 순서, 같은 URL 은 한 번만 넣는다.
    """
    if not messages:
        return []
    titles: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            for match in _LISTED.finditer(message.text):
                titles.setdefault(match['url'], match['title'].strip())
    cited = (url.rstrip('.,;:') for url in _URL.findall(messages[-1].text))
    return [Source(title=titles[url], url=url) for url in dict.fromkeys(cited) if url in titles]
