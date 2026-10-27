"""researcher 도구(tools.py) 시나리오."""

import pytest
from conftest import FakeSearchClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from book_agent.services import VectorStoreService, WebSearchService
from book_agent.tools import build_research_tools

RESULTS = [
    {'title': '파이썬 공식 문서', 'url': 'https://docs.python.org', 'content': '파이썬 설치'},
    {'title': 'PEP 8', 'url': 'https://peps.python.org/pep-0008', 'content': '코딩 스타일'},
]


@pytest.fixture
def make_tools(tmp_path):
    def factory(results=RESULTS, top_k=5):
        vectors = VectorStoreService(
            persist_dir=tmp_path / 'chroma',
            embeddings=DeterministicFakeEmbedding(size=16),
            collection_name='test',
        )
        web = WebSearchService(FakeSearchClient(results), tmp_path / 'res')
        return build_research_tools(web, vectors, top_k=top_k)

    return factory


def test_tools_expose_single_query_argument(make_tools):
    """LLM 에는 query 인자 하나만 노출되고, 설명(docstring)이 붙는다.

    대상: tools.build_research_tools
    구분: 정상
    우선순위: 상
    사전조건: 가짜 검색 클라이언트, 가짜 임베딩
    절차: 도구 목록의 name/args/description 확인
    기대결과: [web_search, retrieve], 각 args=['query'], 설명 비어 있지 않음
    """
    tools = make_tools()
    assert [t.name for t in tools] == ['web_search', 'retrieve']
    for t in tools:
        assert list(t.args) == ['query']
        assert t.description


def test_web_search_reports_count_and_listing(make_tools):
    """web_search 는 검색 건수, 새 청크 수, 제목(URL) 목록을 보고한다.

    대상: tools.web_search
    구분: 정상
    우선순위: 상
    사전조건: 검색 결과 2건
    절차: web_search.invoke({'query': '파이썬'})
    기대결과: '2건 검색, 새 청크 2개 저장' + 두 문서의 '제목 (URL)'
    """
    web_search, _ = make_tools()
    report = web_search.invoke({'query': '파이썬'})
    assert report.startswith('2건 검색, 새 청크 2개 저장')
    assert '- 파이썬 공식 문서 (https://docs.python.org)' in report
    assert '- PEP 8 (https://peps.python.org/pep-0008)' in report


def test_web_search_twice_reports_no_new_chunks(make_tools):
    """같은 검색을 반복하면 새로 저장되는 청크가 0 으로 보고된다.

    대상: tools.web_search
    구분: 경계
    우선순위: 중
    사전조건: 같은 결과를 돌려주는 검색 클라이언트
    절차: 같은 query 로 2회 호출
    기대결과: 2회차 보고에 '새 청크 0개'
    """
    web_search, _ = make_tools()
    web_search.invoke({'query': 'q'})
    assert '새 청크 0개' in web_search.invoke({'query': 'q'})


def test_web_search_with_no_results(make_tools):
    """검색 결과가 없으면 0건으로 보고한다.

    대상: tools.web_search
    구분: 경계
    우선순위: 중
    사전조건: 결과 0건
    절차: web_search.invoke
    기대결과: '0건 검색, 새 청크 0개 저장'
    """
    web_search, _ = make_tools(results=[])
    assert web_search.invoke({'query': 'q'}).startswith('0건 검색, 새 청크 0개 저장')


def test_retrieve_finds_what_web_search_stored(make_tools):
    """web_search 로 저장한 문서를 retrieve 로 [번호] 제목 (출처) 형식으로 찾는다.

    대상: tools.retrieve
    구분: 정상
    우선순위: 상
    사전조건: web_search 로 2건 저장
    절차: retrieve.invoke({'query': '파이썬'})
    기대결과: [1], [2] 번호와 출처 URL 포함
    """
    web_search, retrieve = make_tools()
    web_search.invoke({'query': '파이썬'})
    found = retrieve.invoke({'query': '파이썬'})
    assert found.startswith('[1] ') and '\n\n[2] ' in found
    assert 'https://docs.python.org' in found


def test_retrieve_respects_top_k(make_tools):
    """retrieve 결과 수는 top_k 로 제한된다.

    대상: tools.retrieve
    구분: 경계
    우선순위: 중
    사전조건: top_k=1, 문서 2건 저장
    절차: retrieve.invoke
    기대결과: [1] 만 있고 [2] 없음
    """
    web_search, retrieve = make_tools(top_k=1)
    web_search.invoke({'query': 'q'})
    found = retrieve.invoke({'query': '파이썬'})
    assert '[1] ' in found and '[2] ' not in found


def test_retrieve_on_empty_store(make_tools):
    """저장된 자료가 없으면 안내 문구를 돌려준다.

    대상: tools.retrieve
    구분: 경계
    우선순위: 중
    사전조건: 벡터 DB 비어 있음
    절차: retrieve.invoke
    기대결과: '검색 결과가 없습니다.'
    """
    _, retrieve = make_tools()
    assert retrieve.invoke({'query': 'q'}) == '검색 결과가 없습니다.'
