"""
CLI 통합 시나리오. 실제 `book-agent` 명령을 별도 프로세스로 실행한다(`pytest --live`).

사용자가 쓰는 방식 그대로 표준입력으로 대화를 넣고, 프로세스가 끝난 뒤
디스크에 남은 결과(목차, 벡터 DB, 검색 기록, SQLite 체크포인트)를 확인한다.
DATA_DIR 환경변수로 저장 위치를 임시 폴더로 바꾸므로 실제 data/ 폴더는 건드리지 않는다.
입출력은 파이프이지만 앱이 UTF-8 로 고정하므로 UTF-8 로 주고받는다(app.use_utf8_io).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from book_agent.app import open_checkpointer
from book_agent.graph import build_graph
from book_agent.settings import load_settings

pytestmark = pytest.mark.live

CLI = Path(sys.executable).parent / ('book-agent.exe' if os.name == 'nt' else 'book-agent')
ENCODING = 'utf-8'
STARTUP_BUDGET = 10  # 프로세스 시작과 라이브러리 import 에 드는 시간(초) 여유

FIRST_REQUEST = (
    '초보자용 파이썬 입문서 목차를 4장으로 만들어 줘. '
    '최신 파이썬 버전 정보만 한 번 조사하고, 질문 없이 바로 목차를 작성해.'
)
EDIT_REQUEST = '마지막에 「실전 프로젝트」 장을 하나 추가해 줘. 조사나 질문 없이 바로 반영해.'


def run_cli(inputs, data_dir, thread='book_001', timeout=600, **env_overrides):
    """book-agent 를 실행하고 inputs 를 한 줄씩 입력한다."""
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONIOENCODING'}
    env.update({'DATA_DIR': str(data_dir), 'THREAD_ID': thread, **env_overrides})
    return subprocess.run(
        [str(CLI)],
        input='\n'.join(inputs) + '\n',
        capture_output=True,
        text=True,
        encoding=ENCODING,
        errors='strict',  # UTF-8 이 아닌 출력이 섞이면 바로 실패한다
        env=env,
        cwd=data_dir,
        timeout=timeout,
    )


def saved_state(data_dir, thread='book_001'):
    """프로세스가 남긴 SQLite 체크포인트에서 thread 의 마지막 상태를 읽는다."""
    with open_checkpointer(data_dir / 'checkpoints.sqlite') as saver:
        return build_graph(checkpointer=saver).get_state({'configurable': {'thread_id': thread}})


@pytest.fixture(scope='module')
def session_dir(tmp_path_factory):
    """INT-03, INT-04 가 함께 쓰는 저장 위치(프로그램을 껐다 켜도 같은 폴더)."""
    return tmp_path_factory.mktemp('cli_session')


def test_cli_starts_and_exits(tmp_path):
    """book-agent 명령이 시작 안내를 출력하고 'q' 로 정상 종료한다(API 호출 없음).

    대상: book-agent 콘솔 스크립트 (book_agent:main → app.main)
    구분: 정상
    우선순위: 상
    사전조건: 빈 DATA_DIR, 실제 .env
    절차: book-agent 실행 → 'q' 입력
    기대결과: 종료 코드 0, thread/목차 경로 안내와 'Goodbye!!' 출력,
        checkpoints.sqlite 만 생성(목차·벡터 DB 미생성)
    """
    result = run_cli(['q'], tmp_path)

    assert result.returncode == 0, result.stderr
    assert 'AI 목차 작성 도우미' in result.stdout  # 한글이 UTF-8 로 출력된다
    assert 'thread: book_001' in result.stdout
    assert str(tmp_path / 'outline.md') in result.stdout
    assert 'Goodbye!!' in result.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoints.sqlite']


def test_cli_ignores_blank_input(tmp_path):
    """빈 줄과 공백만 있는 입력은 무시하고, 대화 상태도 만들지 않는다(API 호출 없음).

    대상: app.main 입력 처리
    구분: 경계
    우선순위: 중
    사전조건: 빈 DATA_DIR
    절차: '', '   ' 입력 후 'exit'
    기대결과: 종료 코드 0, 체크포인트에 해당 thread 상태 없음
    """
    result = run_cli(['', '   ', 'exit'], tmp_path)

    assert result.returncode == 0, result.stderr
    assert saved_state(tmp_path).values == {}


def test_cli_first_session_creates_outline(session_dir):
    """첫 세션에서 조사 → 목차 작성 → 보고까지 진행하고 결과를 디스크에 남긴다.

    대상: book-agent 전체 (실제 OpenAI/Tavily, SQLite, Chroma, 파일 저장)
    구분: 정상
    우선순위: 상
    사전조건: 빈 DATA_DIR, 조사 1회와 4장 목차를 요청
    절차: 요청 1건 입력 → 'q'
    기대결과: 종료 코드 0, 인코딩 오류 없음, [supervisor]/[communicator] 출력,
        outline.md 에 '#' 제목, resources/*.json·chroma 생성, 체크포인트는 human 대기·모든 작업 done
    """
    result = run_cli([FIRST_REQUEST, 'q'], session_dir)

    assert result.returncode == 0, result.stderr[-2000:]
    assert 'UnicodeEncodeError' not in result.stderr
    assert '[supervisor]' in result.stdout and '[communicator]' in result.stdout

    outline = (session_dir / 'outline.md').read_text(encoding='utf-8')
    assert '#' in outline
    assert list((session_dir / 'resources').glob('*.json'))
    assert (session_dir / 'chroma').is_dir()

    state = saved_state(session_dir)
    assert state.next == ('human',)
    assert all(t.status == 'done' for t in state.values['tasks'])
    assert state.values['messages'][0].content == FIRST_REQUEST
    assert len(state.values['research_notes']) >= 1


def test_cli_restart_resumes_conversation(session_dir):
    """프로그램을 다시 실행하면 같은 thread 의 대화를 이어 가고 목차를 수정한다.

    대상: SQLite 체크포인트 복원 + Command(resume) (프로세스 재시작)
    구분: 정상
    우선순위: 상
    사전조건: INT-03 이 끝난 같은 DATA_DIR(이전 대화가 human 에서 대기)
    절차: 새 프로세스로 수정 요청 1건 입력 → 'q'
    기대결과: 종료 코드 0, 첫 요청이 대화 맨 앞에 유지, 수정 요청이 대화에 추가,
        목차에 '실전 프로젝트' 포함, 조사 노트 유지, 다시 human 대기
    """
    before = saved_state(session_dir)
    assert before.next == ('human',), '사전조건: INT-03 이 먼저 성공해야 한다'
    notes_before = len(before.values['research_notes'])

    result = run_cli([EDIT_REQUEST, 'q'], session_dir)

    assert result.returncode == 0, result.stderr[-2000:]
    assert 'UnicodeEncodeError' not in result.stderr
    state = saved_state(session_dir)
    humans = [m.content for m in state.values['messages'] if m.type == 'human']
    assert humans[0] == FIRST_REQUEST
    assert humans[-1] == EDIT_REQUEST
    assert len(state.values['messages']) > len(before.values['messages'])
    assert len(state.values['research_notes']) >= notes_before
    assert '실전 프로젝트' in (session_dir / 'outline.md').read_text(encoding='utf-8')
    assert state.next == ('human',)


def test_cli_other_thread_is_separate(session_dir):
    """THREAD_ID 를 바꿔 실행하면 새 대화로 시작하고 기존 대화는 그대로 남는다(API 호출 없음).

    대상: settings.thread_id + 체크포인트 thread 분리
    구분: 정상
    우선순위: 중
    사전조건: INT-03/04 의 DATA_DIR(book_001 대화 존재)
    절차: THREAD_ID=book_002 로 실행 → 'q'
    기대결과: 'thread: book_002' 안내, book_002 상태 없음, book_001 대화 유지
    """
    messages_before = len(saved_state(session_dir).values['messages'])

    result = run_cli(['q'], session_dir, thread='book_002')

    assert result.returncode == 0, result.stderr
    assert 'thread: book_002' in result.stdout
    assert saved_state(session_dir, 'book_002').values == {}
    assert len(saved_state(session_dir).values['messages']) == messages_before


def test_cli_invalid_api_key(tmp_path):
    """OpenAI 키가 잘못되면 재시도 없이 안내 문구를 보여주고 종료 코드 1 로 끝난다.

    대상: app.main 오류 처리 (OpenAI 401, fatal)
    구분: 예외
    우선순위: 상
    사전조건: OPENAI_API_KEY=sk-invalid 환경변수(.env 보다 우선)
    절차: 요청 1건 입력
    기대결과: 종료 코드 1, stderr 에 '[오류] OpenAI API 키가 올바르지 않습니다' 안내,
        traceback 미출력, 5초 안에 종료(재시도 안 함), API 키 원문 미출력
    """
    import time

    start = time.perf_counter()
    result = run_cli(['안녕', 'q'], tmp_path, OPENAI_API_KEY='sk-invalid-key-for-test')
    elapsed = time.perf_counter() - start

    assert result.returncode == 1
    assert '[오류] OpenAI API 키가 올바르지 않습니다' in result.stderr
    assert 'Traceback' not in result.stderr
    assert elapsed < 5 + STARTUP_BUDGET
    output = result.stdout + result.stderr
    assert 'sk-invalid-key-for-test' not in output
    assert load_settings().tavily_api_key.get_secret_value() not in output


def test_cli_invalid_tavily_key(tmp_path):
    """Tavily 키가 잘못되면 조사 단계에서 안내 문구와 함께 종료 코드 1 로 끝난다.

    대상: app.main 오류 처리 (Tavily InvalidAPIKeyError, researcher 도구 안에서 발생)
    구분: 예외
    우선순위: 중
    사전조건: TAVILY_API_KEY=tvly-invalid 환경변수, 실제 OpenAI 키, 빈 벡터 DB
    절차: 웹 검색이 꼭 필요한 요청 1건 입력
    기대결과: 종료 코드 1, stderr 에 TAVILY_API_KEY 확인 안내, traceback 미출력
    """
    request = '오늘 날짜 기준 최신 파이썬 버전을 웹에서 조사해 줘. 질문 없이 바로 조사해.'
    result = run_cli([request, 'q'], tmp_path, TAVILY_API_KEY='tvly-invalid-key-for-test')

    assert result.returncode == 1, result.stdout[-1500:]
    assert 'TAVILY_API_KEY' in result.stderr
    assert 'Traceback' not in result.stderr
    assert 'tvly-invalid-key-for-test' not in result.stdout + result.stderr


def test_cli_invalid_setting_value(tmp_path):
    """설정값이 잘못되면 시작 시점에 항목 이름을 안내하고 종료 코드 1 로 끝난다(API 호출 없음).

    대상: app.main (설정 검증)
    구분: 예외
    우선순위: 중
    사전조건: SEARCH_TOP_K=0 환경변수
    절차: book-agent 실행
    기대결과: 종료 코드 1, stderr 에 '[설정 오류]' 와 SEARCH_TOP_K, traceback 미출력
    """
    result = run_cli(['q'], tmp_path, SEARCH_TOP_K='0')

    assert result.returncode == 1
    assert '[설정 오류]' in result.stderr and 'SEARCH_TOP_K' in result.stderr
    assert 'Traceback' not in result.stderr


def test_cli_stdin_closed_without_quit(tmp_path):
    """'q' 없이 표준입력이 끝나도 오류 없이 종료한다(API 호출 없음).

    대상: app.main (EOFError 처리)
    구분: 예외
    우선순위: 하
    사전조건: 빈 DATA_DIR
    절차: 아무 입력 없이 표준입력 종료
    기대결과: 종료 코드 0, 'Goodbye!!', traceback 미출력
    """
    result = run_cli([], tmp_path)

    assert result.returncode == 0, result.stderr
    assert 'Goodbye!!' in result.stdout
    assert 'Traceback' not in result.stderr
