"""목차 저장소(repositories.py) 시나리오."""

from book_agent.repositories import EMPTY_OUTLINE, FileOutlineRepository, OutlineRepository


def test_load_without_file_returns_placeholder(tmp_path):
    """목차 파일이 아직 없으면 안내 문구를 돌려주고 파일을 만들지 않는다.

    대상: repositories.FileOutlineRepository.load
    구분: 예외
    우선순위: 상
    사전조건: outline.md 없음
    절차: load() 호출
    기대결과: EMPTY_OUTLINE 반환, 파일은 여전히 없음
    """
    path = tmp_path / 'outline.md'
    assert FileOutlineRepository(path).load() == EMPTY_OUTLINE
    assert not path.exists()


def test_save_creates_parent_dirs_and_roundtrips(tmp_path):
    """저장 시 상위 폴더를 만들고, 한글 목차가 그대로 다시 읽힌다.

    대상: repositories.FileOutlineRepository.save/load
    구분: 정상
    우선순위: 상
    사전조건: 상위 폴더 nested/dir 없음
    절차: 1) 한글 마크다운 목차 save 2) load
    기대결과: 폴더 생성, 저장한 문자열과 동일하게 읽힘
    """
    path = tmp_path / 'nested' / 'dir' / 'outline.md'
    repo = FileOutlineRepository(path)

    repo.save('# 1장 파이썬 소개\n## 1.1 설치')

    assert path.exists()
    assert repo.load() == '# 1장 파이썬 소개\n## 1.1 설치'


def test_save_writes_utf8(tmp_path):
    """파일은 UTF-8 로 저장되어 Windows(cp949)에서도 다른 도구로 열 수 있다.

    대상: repositories.FileOutlineRepository.save
    구분: 정상
    우선순위: 중
    사전조건: 없음
    절차: 한글/이모지 포함 목차 저장 후 바이트를 UTF-8 로 디코딩
    기대결과: 원문과 동일
    """
    path = tmp_path / 'outline.md'
    FileOutlineRepository(path).save('# 한글 목차 ✅')
    assert path.read_bytes().decode('utf-8') == '# 한글 목차 ✅'


def test_save_overwrites(tmp_path):
    """다시 저장하면 이전 목차를 덮어쓴다.

    대상: repositories.FileOutlineRepository.save
    구분: 정상
    우선순위: 중
    사전조건: 'old' 저장됨
    절차: 'new' 저장 후 load
    기대결과: 'new'
    """
    repo = FileOutlineRepository(tmp_path / 'outline.md')
    repo.save('old')
    repo.save('new')
    assert repo.load() == 'new'


def test_empty_outline_roundtrip(tmp_path):
    """빈 문자열도 그대로 저장/복원된다(안내 문구로 바뀌지 않음).

    대상: repositories.FileOutlineRepository
    구분: 경계
    우선순위: 하
    사전조건: 없음
    절차: '' 저장 후 load
    기대결과: ''
    """
    repo = FileOutlineRepository(tmp_path / 'outline.md')
    repo.save('')
    assert repo.load() == ''


def test_file_repository_satisfies_protocol(tmp_path):
    """파일 저장소는 OutlineRepository 프로토콜을 만족해 노드에 주입할 수 있다.

    대상: repositories.OutlineRepository
    구분: 정상
    우선순위: 하
    사전조건: 없음
    절차: 프로토콜 타입 변수에 대입 후 load/save 존재 확인
    기대결과: load, save 호출 가능
    """
    repo: OutlineRepository = FileOutlineRepository(tmp_path / 'o.md')
    assert callable(repo.load) and callable(repo.save)
