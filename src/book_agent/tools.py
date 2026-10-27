"""
researcher  에이전트가 사용하는 도구 모듈.

도구는 서비스 객체를 클로저로 캡처한다.
전역 변수 없이도 LLM에는 'query' 인자 하나만 노출된다.
"""

from langchain_core.tools import BaseTool, tool

from .services import VectorStoreService, WebSearchService, format_documents


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
