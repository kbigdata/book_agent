import hashlib
import json
import logging
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .device import resolve_device
from .settings import Settings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- 임베딩
def build_embeddings(settings: Settings) -> Embeddings:
    """설정에 따라 OpenAI(API)  또는 HuggingFace(로컬 GPU) 임베딩을 만든다."""
    if settings.embedding_provider == 'huggingface':
        # 선택 의존성(pyproject 의 gpu 그룹)이므로 필요할 때만 import 한다.
        from langchain_huggingface import HuggingFaceEmbeddings

        device = resolve_device(settings.device)
        logger.info(f'로컬 임베딩 {settings.hf_embedding_model}를 {device}에서 실행합니다.')

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)


# 벡터 DB : 결정적 id로 중복 제거
def document_id(doc: Document) -> str:
    """출처 URL + 본문으로 만든 결정적 ID. 같은 청크는 항상 같은 ID를 갖는다."""
    key = f'{doc.metadata.get("source", "")}\n{doc.page_content}'

    return hashlib.sha256(key.encode('utf-8')).hexdigest()


class VectorStoreService:
    """Chroma DB 래퍼, 청크 분할, 중복 제거, 검색을 담당한다."""

    def __init__(
        self,
        persist_dir: Path,
        embeddings: Embeddings,
        collection_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    @cached_property
    def store(self) -> Chroma:
        # 처음 접근할 때 한 번만 DB를 연다(지연 초기화)
        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )

    def add_documents(self, documents: list[Document]) -> int:
        """문서를 청크로 나눠 저장하고, 새로 저장한 청크 수를 반환한다."""
        chunks = {document_id(c): c for c in self._splitter.split_documents(documents)}

        if not chunks:
            return 0

        # 이미 있는 ID만 조회한다. 전체 컬렉션을 읽지 않으므로 DB가 커져도 빠르다.
        existing = set(self.store.get(ids=list(chunks), include=[])['ids'])

        new_ids = [i for i in chunks if i not in existing]

        if new_ids:
            # 새 청크만 임베딩하므로 임베딩 비용도 중복으로 들지 않는다.
            self.store.add_documents([chunks[i] for i in new_ids], ids=new_ids)
        logger.info(f'청크 {len(chunks)}개 중 {len(new_ids)}개를 새로 저장했습니다.')
        return len(new_ids)

    def search(self, query: str, k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, k=k)


# 웹 검색
class SearchClient(Protocol):
    """TavilyClient 가 만족하는 최소 인터페이스, 테스트에서는 가짜 객체로 바꾼다."""

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


class WebSearchService:
    """웹 검색 결과를 json으로 보관하고 langchain document로 변환한다."""

    def __init__(self, client: SearchClient, save_dir: Path, max_results: int = 5) -> None:
        self._client = client
        self._save_dir = save_dir
        self._max_results = max_results

    def search(self, query: str) -> list[Document]:
        response = self._client.search(
            query=query,
            search_depth='advanced',
            include_raw_content=True,
            max_results=self._max_results,
        )
        results = response.get('results', [])
        self._save(query, results)
        return [self._to_document(r) for r in results]

    def _save(self, query: str, results: list[dict[str, Any]]) -> Path:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        # 초 단위 시각 + uuid -> 병렬 검색에서도 파일명이 겹치지 않는다.
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self._save_dir / f'{stamp}_{uuid4().hex[:8]}.json'
        payload = {'query': query, 'results': results}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    @staticmethod
    def _to_document(result: dict[str, Any]) -> Document:
        # 본문(raw_content)과 요약(content) 중 더 긴 쪽을 사용한다.
        raw = result.get('raw_content') or ''
        summary = result.get('content') or ''

        return Document(
            page_content=raw if len(raw) > len(summary) else summary,
            metadata={'title': result.get('title', ''), 'source': result.get('url', '')},
        )


def format_documents(docs: list[Document], max_chars: int = 500) -> str:
    """프롬프트에 넣을 수 있도록 문서를 [번호] 제목(출처) + 발췌 형태로 만든다."""
    if not docs:
        return '검색 결과가 없습니다.'

    return '\n\n'.join(
        f'[{i}] {d.metadata.get("title", "")} ({d.metadata.get("source", "")})\n'
        f'{d.page_content[:max_chars]}'
        for i, d in enumerate(docs, 1)
    )
