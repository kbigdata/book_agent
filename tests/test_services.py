"""서비스(services.py) 시나리오. 웹/임베딩 API 는 가짜로 대체한다."""

import json
import sys
from types import SimpleNamespace

import pytest
from conftest import FakeSearchClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from book_agent.services import (
    VectorStoreService,
    WebSearchService,
    build_embeddings,
    document_id,
    format_documents,
)

RESULTS = [
    {
        'title': '파이썬 공식 문서',
        'url': 'https://docs.python.org',
        'content': '요약',
        'raw_content': '아주 긴 본문 ' * 20,
    },
    {
        'title': '짧은 글',
        'url': 'https://short.example',
        'content': '요약만 있음',
        'raw_content': None,
    },
]


@pytest.fixture
def make_vectors(tmp_path):
    def factory(collection='test', persist_dir=None):
        return VectorStoreService(
            persist_dir=persist_dir or tmp_path / 'chroma',
            embeddings=DeterministicFakeEmbedding(size=16),
            collection_name=collection,
            chunk_size=200,
            chunk_overlap=0,
        )

    return factory


@pytest.fixture
def vectors(make_vectors):
    return make_vectors()


# ------------------------------------------------------------------ 웹 검색
def test_web_search_passes_options_and_saves_json(tmp_path):
    """검색 옵션을 클라이언트에 넘기고, 원본 결과를 JSON 파일로 남긴다.

    대상: services.WebSearchService.search
    구분: 정상
    우선순위: 상
    사전조건: 결과 2건을 돌려주는 가짜 Tavily 클라이언트, max_results=3
    절차: search('파이썬 책')
    기대결과: advanced/raw_content/max_results=3 으로 1회 호출, JSON 1개 저장(query+results),
        Document 2개
    """
    client = FakeSearchClient(RESULTS)
    web = WebSearchService(client=client, save_dir=tmp_path / 'res', max_results=3)

    docs = web.search('파이썬 책')

    assert client.calls == [
        {
            'query': '파이썬 책',
            'search_depth': 'advanced',
            'include_raw_content': True,
            'max_results': 3,
        }
    ]
    saved = list((tmp_path / 'res').glob('*.json'))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding='utf-8'))
    assert payload['query'] == '파이썬 책' and len(payload['results']) == 2
    assert len(docs) == 2


def test_web_search_default_max_results(tmp_path):
    """max_results 를 지정하지 않으면 5건을 요청한다.

    대상: services.WebSearchService
    구분: 정상
    우선순위: 하
    사전조건: 기본 생성자
    절차: search('q')
    기대결과: max_results=5 로 호출
    """
    client = FakeSearchClient([])
    WebSearchService(client=client, save_dir=tmp_path).search('q')
    assert client.calls[0]['max_results'] == 5


def test_web_search_uses_longer_text_and_keeps_metadata(tmp_path):
    """본문과 요약 중 긴 쪽을 쓰고, raw_content 가 None 이면 요약을 쓴다.

    대상: services.WebSearchService._to_document
    구분: 정상
    우선순위: 상
    사전조건: 1번 결과는 본문이 길고, 2번 결과는 raw_content=None
    절차: search('q')
    기대결과: 1번은 본문, 2번은 요약 사용, metadata 에 title/source(url)
    """
    docs = WebSearchService(FakeSearchClient(RESULTS), tmp_path).search('q')

    assert docs[0].page_content.startswith('아주 긴 본문')
    assert docs[0].metadata == {'title': '파이썬 공식 문서', 'source': 'https://docs.python.org'}
    assert docs[1].page_content == '요약만 있음'


def test_web_search_missing_fields_become_empty(tmp_path):
    """title/url/content 가 빠진 결과도 빈 문자열로 처리되어 오류가 나지 않는다.

    대상: services.WebSearchService._to_document
    구분: 예외
    우선순위: 중
    사전조건: 빈 dict 결과 1건
    절차: search('q')
    기대결과: page_content='', metadata={'title': '', 'source': ''}
    """
    [doc] = WebSearchService(FakeSearchClient([{}]), tmp_path).search('q')
    assert doc.page_content == ''
    assert doc.metadata == {'title': '', 'source': ''}


def test_web_search_handles_empty_response(tmp_path):
    """검색 결과가 없어도 빈 목록을 돌려주고 기록 파일은 남긴다.

    대상: services.WebSearchService.search
    구분: 경계
    우선순위: 중
    사전조건: results=[] 응답
    절차: search('q')
    기대결과: [] 반환, JSON 1개(results=[]) 저장
    """
    assert WebSearchService(FakeSearchClient([]), tmp_path).search('q') == []
    [saved] = tmp_path.glob('*.json')
    assert json.loads(saved.read_text(encoding='utf-8'))['results'] == []


def test_web_search_files_do_not_collide(tmp_path):
    """같은 초에 여러 번 검색해도 기록 파일 이름이 겹치지 않는다.

    대상: services.WebSearchService._save
    구분: 경계
    우선순위: 중
    사전조건: 없음
    절차: 연속으로 3회 검색
    기대결과: JSON 파일 3개
    """
    web = WebSearchService(FakeSearchClient(RESULTS), tmp_path)
    for _ in range(3):
        web.search('q')
    assert len(list(tmp_path.glob('*.json'))) == 3


# ------------------------------------------------------------------ 벡터 DB
def test_document_id_is_deterministic():
    """같은 출처+본문은 항상 같은 id, 출처나 본문이 다르면 다른 id.

    대상: services.document_id
    구분: 정상
    우선순위: 상
    사전조건: 없음
    절차: 같은/다른 문서의 id 비교
    기대결과: 같으면 동일, 출처·본문 중 하나라도 다르면 상이, 64자리 sha256
    """
    a = Document('본문', metadata={'source': 'u1'})
    assert document_id(a) == document_id(Document('본문', metadata={'source': 'u1'}))
    assert document_id(a) != document_id(Document('본문', metadata={'source': 'u2'}))
    assert document_id(a) != document_id(Document('다른 본문', metadata={'source': 'u1'}))
    assert len(document_id(a)) == 64


def test_document_id_without_source():
    """source 가 없는 문서도 id 를 만들 수 있다.

    대상: services.document_id
    구분: 예외
    우선순위: 하
    사전조건: metadata 없음
    절차: document_id(Document('본문'))
    기대결과: 오류 없이 64자리 id
    """
    assert len(document_id(Document('본문'))) == 64


def test_store_is_opened_lazily(vectors):
    """Chroma 는 처음 사용할 때 한 번만 열린다.

    대상: services.VectorStoreService.store
    구분: 정상
    우선순위: 중
    사전조건: 서비스 생성 직후
    절차: store 접근 전/후 상태 비교, 두 번 접근
    기대결과: 접근 전에는 미생성, 두 번 접근해도 같은 객체
    """
    assert 'store' not in vars(vectors)
    assert vectors.store is vectors.store


def test_add_documents_skips_duplicates(vectors):
    """같은 문서를 두 번 넣으면 두 번째에는 새로 저장되는 청크가 없다.

    대상: services.VectorStoreService.add_documents
    구분: 정상
    우선순위: 상
    사전조건: chunk_size=200, 약 600자 문서
    절차: 같은 문서로 add_documents 2회
    기대결과: 1회차 여러 청크 저장, 2회차 0, 컬렉션 청크 수 변화 없음
    """
    text = ' '.join(f'문장{i}' for i in range(100))
    docs = [Document(text, metadata={'source': 'u', 'title': 't'})]

    first = vectors.add_documents(docs)
    second = vectors.add_documents(docs)

    assert first > 1  # chunk_size=200 이므로 여러 청크로 나뉜다.
    assert second == 0
    assert len(vectors.store.get()['ids']) == first


def test_identical_chunks_in_one_call_are_stored_once(vectors):
    """한 번의 호출 안에서 내용이 같은 청크도 하나만 저장된다.

    대상: services.VectorStoreService.add_documents
    구분: 경계
    우선순위: 중
    사전조건: '가'*450 문서 → 200+200+50 자로 분할(앞 두 청크 동일)
    절차: add_documents 1회
    기대결과: 2개 저장
    """
    docs = [Document('가' * 450, metadata={'source': 'u', 'title': 't'})]
    assert vectors.add_documents(docs) == 2


def test_add_empty_documents_returns_zero(vectors):
    """빈 목록은 DB 를 건드리지 않고 0 을 돌려준다.

    대상: services.VectorStoreService.add_documents
    구분: 경계
    우선순위: 중
    사전조건: 없음
    절차: add_documents([])
    기대결과: 0 반환, store 미생성(지연 초기화 유지)
    """
    assert vectors.add_documents([]) == 0
    assert 'store' not in vars(vectors)


def test_search_returns_saved_documents(vectors):
    """저장한 문서를 검색으로 찾고, 메타데이터가 유지된다.

    대상: services.VectorStoreService.search
    구분: 정상
    우선순위: 상
    사전조건: 문서 1건 저장
    절차: search('파이썬', k=1)
    기대결과: 저장한 본문과 metadata(title, source)
    """
    vectors.add_documents([Document('파이썬 설치 방법', metadata={'source': 'u', 'title': 't'})])
    [found] = vectors.search('파이썬', k=1)
    assert found.page_content == '파이썬 설치 방법'
    assert found.metadata == {'source': 'u', 'title': 't'}


def test_search_respects_k(vectors):
    """검색 결과 수는 k 를 넘지 않는다.

    대상: services.VectorStoreService.search
    구분: 경계
    우선순위: 중
    사전조건: 서로 다른 문서 5건 저장
    절차: search(k=2), search(k=10)
    기대결과: 2건, 5건(저장된 수가 상한)
    """
    vectors.add_documents(
        [Document(f'문서 {i}', metadata={'source': f'u{i}', 'title': 't'}) for i in range(5)]
    )
    assert len(vectors.search('문서', k=2)) == 2
    assert len(vectors.search('문서', k=10)) == 5


def test_search_on_empty_store(vectors):
    """비어 있는 DB 를 검색하면 빈 목록을 돌려준다.

    대상: services.VectorStoreService.search
    구분: 경계
    우선순위: 중
    사전조건: 저장된 문서 없음
    절차: search('q')
    기대결과: []
    """
    assert vectors.search('q') == []


def test_store_persists_across_instances(make_vectors):
    """같은 폴더/컬렉션으로 다시 열면 이전에 저장한 문서가 남아 있다.

    대상: services.VectorStoreService (persist_directory)
    구분: 정상
    우선순위: 상
    사전조건: 첫 인스턴스로 문서 저장
    절차: 같은 경로로 새 인스턴스 생성 후 검색, 같은 문서 재저장
    기대결과: 문서 검색됨, 재저장 시 새 청크 0
    """
    doc = Document('영속 저장 확인', metadata={'source': 'u', 'title': 't'})
    make_vectors().add_documents([doc])

    reopened = make_vectors()
    assert [d.page_content for d in reopened.search('영속', k=1)] == ['영속 저장 확인']
    assert reopened.add_documents([doc]) == 0


def test_collections_are_isolated(make_vectors):
    """임베딩 제공자별 컬렉션은 서로 섞이지 않는다.

    대상: services.VectorStoreService(collection_name)
    구분: 정상
    우선순위: 중
    사전조건: web_openai 컬렉션에만 문서 저장
    절차: web_huggingface 컬렉션에서 검색
    기대결과: []
    """
    make_vectors('web_openai').add_documents([Document('A', metadata={'source': 'u'})])
    assert make_vectors('web_huggingface').search('A') == []


# ------------------------------------------------------------------ 포맷
def test_format_documents_numbers_title_source_and_truncates():
    """[번호] 제목 (출처) + 발췌 형식으로 만들고, 발췌는 max_chars 로 자른다.

    대상: services.format_documents
    구분: 정상
    우선순위: 상
    사전조건: 문서 2건
    절차: format_documents(docs, max_chars=3)
    기대결과: '[1] A (https://a)\\nxxx\\n\\n[2] B (https://b)\\ny'
    """
    docs = [
        Document('x' * 10, metadata={'title': 'A', 'source': 'https://a'}),
        Document('y', metadata={'title': 'B', 'source': 'https://b'}),
    ]
    text = format_documents(docs, max_chars=3)
    assert text == '[1] A (https://a)\nxxx\n\n[2] B (https://b)\ny'


def test_format_documents_default_limit_is_500():
    """기본 발췌 길이는 500자다.

    대상: services.format_documents
    구분: 경계
    우선순위: 하
    사전조건: 본문 800자 문서
    절차: format_documents([doc])
    기대결과: 본문 부분이 정확히 500자
    """
    text = format_documents([Document('z' * 800, metadata={'title': 'T', 'source': 's'})])
    assert text.split('\n', 1)[1] == 'z' * 500


def test_format_documents_without_metadata():
    """제목/출처가 없어도 형식이 깨지지 않는다.

    대상: services.format_documents
    구분: 예외
    우선순위: 하
    사전조건: metadata 없는 문서
    절차: format_documents([Document('본문')])
    기대결과: '[1]  ()\\n본문'
    """
    assert format_documents([Document('본문')]) == '[1]  ()\n본문'


def test_format_documents_empty():
    """문서가 없으면 안내 문구를 돌려준다.

    대상: services.format_documents
    구분: 경계
    우선순위: 중
    사전조건: 없음
    절차: format_documents([])
    기대결과: '검색 결과가 없습니다.'
    """
    assert format_documents([]) == '검색 결과가 없습니다.'


# ------------------------------------------------------------------ 임베딩
def test_build_embeddings_openai_by_default(make_settings):
    """기본 설정에서는 OpenAI 임베딩을 설정된 모델로 만든다(API 호출 없음).

    대상: services.build_embeddings
    구분: 정상
    우선순위: 상
    사전조건: embedding_provider=openai(기본)
    절차: build_embeddings(settings)
    기대결과: OpenAIEmbeddings, model=text-embedding-3-large
    """
    embeddings = build_embeddings(make_settings())
    assert type(embeddings).__name__ == 'OpenAIEmbeddings'
    assert embeddings.model == 'text-embedding-3-large'


def test_build_embeddings_huggingface(make_settings, monkeypatch):
    """huggingface 설정이면 로컬 임베딩을 지정 장치와 정규화 옵션으로 만든다.

    대상: services.build_embeddings
    구분: 정상
    우선순위: 중
    사전조건: langchain_huggingface 를 가짜 모듈로 대체(모델 다운로드 방지), device=cpu
    절차: build_embeddings(settings)
    기대결과: model_name=BAAI/bge-m3, device=cpu, normalize_embeddings=True 로 생성
    """
    created = {}

    def fake_embeddings(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(**kwargs)

    fake_module = SimpleNamespace(HuggingFaceEmbeddings=fake_embeddings)
    monkeypatch.setitem(sys.modules, 'langchain_huggingface', fake_module)

    build_embeddings(make_settings(embedding_provider='huggingface', device='cpu'))

    assert created == {
        'model_name': 'BAAI/bge-m3',
        'model_kwargs': {'device': 'cpu'},
        'encode_kwargs': {'normalize_embeddings': True},
    }


def test_build_embeddings_applies_timeout(make_settings):
    """OpenAI 임베딩에도 LLM 타임아웃과 재시도 횟수가 적용된다.

    대상: services.build_embeddings (settings.llm_timeout, llm_max_retries)
    구분: 정상
    우선순위: 중
    사전조건: llm_timeout=30, llm_max_retries=0
    절차: build_embeddings(settings)
    기대결과: request_timeout=30, max_retries=0
    """
    embeddings = build_embeddings(make_settings(llm_timeout=30, llm_max_retries=0))
    assert (embeddings.request_timeout, embeddings.max_retries) == (30, 0)
