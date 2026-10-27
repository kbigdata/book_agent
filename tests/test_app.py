"""앱(app.py, __init__.py) 시나리오. 조립, 체크포인트, 출력, 대화 루프를 검증한다."""

import sqlite3

import pytest
from conftest import plan
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import book_agent
import book_agent.app as app
from book_agent.app import EXIT_WORDS, build_context, open_checkpointer, run_turn
from book_agent.graph import build_graph


def config(thread='t1'):
    return {'configurable': {'thread_id': thread}, 'recursion_limit': 40}


# ------------------------------------------------------------------ build_context
def test_build_context_wires_dependencies(make_settings):
    """실제 의존성 조립이 API 호출 없이 성공하고 설정값이 전달된다.

    대상: app.build_context
    구분: 정상
    우선순위: 상
    사전조건: 더미 API 키, data_dir=tmp_path, 조사 한도 4, 기록 한도 1234
    절차: build_context(settings)
    기대결과: 한도 값 전달, 목차 저장소 경로 연결, researcher 에이전트 이름, LLM 모델명,
        LLM 타임아웃 120초·재시도 1회 적용(타임아웃 없이 무한 대기하지 않음)
    """
    settings = make_settings(max_research_rounds=4, max_history_tokens=1234, llm_model='m-test')
    ctx = build_context(settings)

    assert ctx.max_research_rounds == 4
    assert ctx.max_history_tokens == 1234
    assert ctx.outline_repo.load() == '아직 작성된 목차가 없습니다.'
    assert ctx.researcher.name == 'researcher'
    assert ctx.llm.model_name == 'm-test'
    assert (ctx.llm.request_timeout, ctx.llm.max_retries) == (120, 1)


def test_build_context_researcher_has_tools(make_settings):
    """researcher 하위 에이전트에 web_search, retrieve 도구가 연결된다.

    대상: app.build_context (create_agent)
    구분: 정상
    우선순위: 상
    사전조건: 더미 API 키
    절차: researcher 그래프의 tools 노드 확인
    기대결과: tools 노드에 web_search, retrieve 등록
    """
    ctx = build_context(make_settings())
    tool_node = ctx.researcher.nodes['tools'].bound
    assert set(tool_node.tools_by_name) == {'web_search', 'retrieve'}


def test_build_context_does_not_touch_disk(make_settings, tmp_path):
    """조립 단계에서는 DB/파일을 만들지 않는다(지연 초기화).

    대상: app.build_context
    구분: 정상
    우선순위: 중
    사전조건: 빈 data_dir
    절차: build_context 후 data_dir 내용 확인
    기대결과: chroma, resources, outline.md 미생성
    """
    build_context(make_settings())
    assert not any(tmp_path.iterdir())


# ------------------------------------------------------------------ open_checkpointer
def test_open_checkpointer_creates_db_and_closes(tmp_path):
    """상위 폴더와 DB 파일을 만들고, with 블록이 끝나면 연결을 닫는다.

    대상: app.open_checkpointer
    구분: 정상
    우선순위: 상
    사전조건: 상위 폴더 없음
    절차: with 블록 진입/종료 후 연결 사용
    기대결과: DB 파일 생성, 블록 종료 후 연결 사용 시 ProgrammingError
    """
    db = tmp_path / 'nested' / 'cp.sqlite'
    with open_checkpointer(db) as saver:
        conn = saver.conn
        conn.execute('select 1')
    assert db.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('select 1')


def test_open_checkpointer_closes_on_error(tmp_path):
    """블록 안에서 예외가 나도 연결은 닫히고 예외는 그대로 전달된다.

    대상: app.open_checkpointer (finally)
    구분: 예외
    우선순위: 중
    사전조건: 없음
    절차: with 블록 안에서 RuntimeError 발생
    기대결과: RuntimeError 전달, 연결 닫힘
    """
    with pytest.raises(RuntimeError), open_checkpointer(tmp_path / 'cp.sqlite') as saver:
        conn = saver.conn
        raise RuntimeError('boom')
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('select 1')


def test_sqlite_checkpoint_restores_custom_types(tmp_path, make_ctx):
    """SQLite 체크포인트를 닫았다 다시 열어도 Task/ResearchNote 가 복원된다.

    대상: app.open_checkpointer + CHECKPOINT_TYPES
    구분: 정상
    우선순위: 상
    사전조건: 1턴 실행 후 체크포인터 닫음
    절차: 같은 DB 를 다시 열어 상태 조회
    기대결과: tasks 는 Task, research_notes 는 ResearchNote 타입으로 복원, human 에서 대기
    """
    db = tmp_path / 'cp.sqlite'
    ctx = make_ctx(plans=[plan('researcher'), plan('content_strategist')], replies=['# 1', '보고'])

    with open_checkpointer(db) as saver:
        build_graph(checkpointer=saver).invoke(
            {'messages': [HumanMessage('책')]}, config(), context=ctx
        )

    with open_checkpointer(db) as saver:
        state = build_graph(checkpointer=saver).get_state(config())

    assert type(state.values['tasks'][0]).__name__ == 'Task'
    assert type(state.values['research_notes'][0]).__name__ == 'ResearchNote'
    assert state.values['tasks'][0].status == 'done'
    assert state.next == ('human',)


def test_resume_after_reopening_checkpoint(tmp_path, make_ctx):
    """프로그램을 다시 켠 것처럼 체크포인트를 새로 열어도 대화를 이어 갈 수 있다.

    대상: app.open_checkpointer + graph 재개
    구분: 정상
    우선순위: 상
    사전조건: 1턴 실행 후 체크포인터 닫음
    절차: 다시 열어 Command(resume) 실행
    기대결과: 이전 메시지 뒤에 새 입력과 답변이 이어짐
    """
    db = tmp_path / 'cp.sqlite'
    ctx = make_ctx(plans=[plan('communicator'), plan('communicator')], replies=['질문', '답변'])
    with open_checkpointer(db) as saver:
        build_graph(checkpointer=saver).invoke(
            {'messages': [HumanMessage('책')]}, config(), context=ctx
        )

    with open_checkpointer(db) as saver:
        graph = build_graph(checkpointer=saver)
        graph.invoke(Command(resume='초보자용'), config(), context=ctx)
        contents = [m.content for m in graph.get_state(config()).values['messages']]

    assert contents == ['책', '질문', '초보자용', '답변']


# ------------------------------------------------------------------ run_turn
def test_run_turn_prints_progress_and_streams_writer_nodes(make_ctx, capsys):
    """진행 상황 요약과 content_strategist/communicator 의 답변이 화면에 나온다.

    대상: app.run_turn
    구분: 정상
    우선순위: 상
    사전조건: 플래너 [researcher, content_strategist], LLM 답변 [목차, 보고]
    절차: run_turn 실행 후 표준출력 확인
    기대결과: supervisor 다음 작업 2줄, researcher 조사 완료(1회차, 출처 0건),
        [content_strategist] 목차, [communicator] 보고
    """
    ctx = make_ctx(
        plans=[plan('researcher', '자료 조사'), plan('content_strategist', '목차 작성')],
        replies=['# 1장', '완료했습니다'],
    )
    graph = build_graph(checkpointer=InMemorySaver())

    run_turn(graph, {'messages': [HumanMessage('책')]}, config(), ctx)

    out = capsys.readouterr().out
    assert '[supervisor] 다음 작업 -> researcher: 자료 조사' in out
    assert '[supervisor] 다음 작업 -> content_strategist: 목차 작성' in out
    assert '[researcher] 조사 완료 (1회차, 출처 0건)' in out
    assert '[content_strategist]\n# 1장' in out
    assert '[communicator]\n완료했습니다' in out


def test_run_turn_hides_researcher_tokens(make_ctx, capsys):
    """researcher 하위 에이전트의 LLM 출력은 화면에 흘리지 않는다.

    대상: app.run_turn (STREAM_NODES)
    구분: 정상
    우선순위: 중
    사전조건: researcher 내부에서 가짜 LLM 이 '비밀 토큰' 생성
    절차: run_turn 실행
    기대결과: '비밀 토큰' 미출력, 조사 완료 요약은 출력
    """
    inner = FakeListChatModel(responses=['비밀 토큰'])
    researcher = RunnableLambda(lambda inp: {'messages': [inner.invoke(inp['messages'])]})
    ctx = make_ctx(plans=[plan('researcher'), plan('communicator')], researcher=researcher)

    run_turn(
        build_graph(checkpointer=InMemorySaver()),
        {'messages': [HumanMessage('책')]},
        config(),
        ctx,
    )

    out = capsys.readouterr().out
    assert '비밀 토큰' not in out
    assert '[researcher] 조사 완료 (1회차, 출처 0건)' in out


def test_run_turn_prints_speaker_header_once(make_ctx, capsys):
    """한 노드가 여러 토큰을 내도 [노드] 머리말은 한 번만 찍힌다.

    대상: app.run_turn (speaking)
    구분: 정상
    우선순위: 하
    사전조건: communicator 답변 1건(여러 토큰으로 스트리밍)
    절차: run_turn 실행
    기대결과: '[communicator]' 1회
    """
    ctx = make_ctx(plans=[plan('communicator')], replies=['여러 글자로 된 답변입니다'])
    run_turn(
        build_graph(checkpointer=InMemorySaver()),
        {'messages': [HumanMessage('책')]},
        config(),
        ctx,
    )
    out = capsys.readouterr().out
    assert out.count('[communicator]') == 1
    assert '여러 글자로 된 답변입니다' in out


# ------------------------------------------------------------------ main / 진입점
@pytest.fixture
def run_main(monkeypatch, make_settings, make_ctx, capsys):
    """
    입력 목록으로 main() 을 실행하고 (run_turn payload 목록, 표준출력) 을 돌려준다.

    ctx    : 쓸 AppContext (기본: 항상 communicator 로 답하는 가짜)
    errors : run_turn 호출 순서별로 던질 예외 목록(None 이면 정상 실행)
    입력이 바닥나면 input() 은 EOFError 를 낸다(실제 표준입력이 끝난 것과 같다).
    """
    settings = make_settings()
    default_ctx = make_ctx(
        plans=[plan('communicator')] * 10, replies=[f'답변{i}' for i in range(10)]
    )
    monkeypatch.setattr(app, 'load_settings', lambda: settings)

    def runner(inputs, ctx=None, errors=()):
        monkeypatch.setattr(app, 'build_context', lambda _: ctx or default_ctx)
        payloads, planned = [], list(errors)

        def spy(graph, payload, cfg, c):
            payloads.append(payload)
            error = planned.pop(0) if planned else None
            if error is not None:
                raise error
            run_turn(graph, payload, cfg, c)

        monkeypatch.setattr(app, 'run_turn', spy)
        queue = iter(inputs)

        def fake_input(_):
            try:
                value = next(queue)
            except StopIteration:
                raise EOFError from None
            if isinstance(value, BaseException):
                raise value
            return value

        monkeypatch.setattr('builtins.input', fake_input)
        app.main()
        captured = capsys.readouterr()
        runner.err = captured.err  # 오류 안내는 stderr 로 나간다
        return payloads, captured.out

    runner.settings = settings
    return runner


def test_main_new_then_resume_then_exit(run_main):
    """빈 입력은 무시하고, 첫 입력은 새 대화, 다음 입력은 resume, 'q' 로 종료한다.

    대상: app.main
    구분: 정상
    우선순위: 상
    사전조건: 가짜 설정/컨텍스트, 입력 ['', '파이썬 책', '좋아요', 'q']
    절차: main() 실행
    기대결과: run_turn 2회(dict, Command(resume='좋아요')), 'Goodbye!!' 출력, 체크포인트 DB 생성
    """
    payloads, out = run_main(['', '파이썬 책', '좋아요', 'q'])

    assert len(payloads) == 2
    assert payloads[0]['messages'][0].content == '파이썬 책'
    assert isinstance(payloads[1], Command) and payloads[1].resume == '좋아요'
    assert 'Goodbye!!' in out
    assert run_main.settings.checkpoint_db.exists()


def test_main_strips_whitespace(run_main):
    """입력 앞뒤 공백은 제거되고, 공백만 있는 입력은 무시된다.

    대상: app.main
    구분: 경계
    우선순위: 하
    사전조건: 입력 ['   ', '  파이썬 책  ', 'q']
    절차: main() 실행
    기대결과: run_turn 1회, 메시지 '파이썬 책'
    """
    payloads, _ = run_main(['   ', '  파이썬 책  ', 'q'])
    assert len(payloads) == 1
    assert payloads[0]['messages'][0].content == '파이썬 책'


@pytest.mark.parametrize('word', ['q', 'quit', 'exit', 'Q', 'EXIT', ' Quit '])
def test_main_exit_words(run_main, word):
    """종료어(q/quit/exit)는 대소문자·공백과 관계없이 바로 종료한다.

    대상: app.main (EXIT_WORDS)
    구분: 경계
    우선순위: 중
    사전조건: 첫 입력이 종료어
    절차: main() 실행
    기대결과: run_turn 0회, 'Goodbye!!'
    """
    payloads, out = run_main([word])
    assert payloads == []
    assert 'Goodbye!!' in out


def test_main_prints_banner(run_main):
    """시작 시 thread_id 와 목차 파일 경로를 안내한다.

    대상: app.main
    구분: 정상
    우선순위: 하
    사전조건: thread_id=book_001(기본)
    절차: main() 실행 후 바로 종료
    기대결과: 'thread: book_001' 과 outline.md 경로 출력
    """
    _, out = run_main(['q'])
    assert 'thread: book_001' in out
    assert str(run_main.settings.outline_path) in out


def test_main_resumes_previous_session(run_main):
    """같은 thread_id 로 다시 실행하면 이전 대화가 human 에서 대기 중이므로 resume 으로 이어 간다.

    대상: app.main (graph.get_state(config).next)
    구분: 정상
    우선순위: 상
    사전조건: 1회차 실행에서 1턴 진행 후 종료
    절차: 2회차 main() 첫 입력 '이어서'
    기대결과: 2회차 첫 payload 가 Command(resume='이어서')
    """
    run_main(['파이썬 책', 'q'])
    payloads, _ = run_main(['이어서', 'q'])
    assert isinstance(payloads[0], Command) and payloads[0].resume == '이어서'


def test_exit_words_constant():
    """종료어 상수는 q, quit, exit 세 가지다.

    대상: app.EXIT_WORDS
    구분: 정상
    우선순위: 하
    사전조건: 없음
    절차: 값 확인
    기대결과: {'q', 'quit', 'exit'}
    """
    assert {'q', 'quit', 'exit'} == EXIT_WORDS


def test_package_main_delegates_to_app(monkeypatch):
    """book-agent 명령(book_agent:main)은 app.main 을 호출한다.

    대상: book_agent.__init__.main
    구분: 정상
    우선순위: 중
    사전조건: app.main 을 호출 기록용 대역으로 교체
    절차: book_agent.main()
    기대결과: app.main 1회 호출
    """
    calls = []
    monkeypatch.setattr(app, 'main', lambda: calls.append('called'))
    book_agent.main()
    assert calls == ['called']


# ------------------------------------------------------------------ 오류 처리
def api_error(cls, status):
    import httpx

    response = httpx.Response(status, request=httpx.Request('POST', 'https://api.openai.com'))
    return cls('detail', response=response, body=None)


def test_main_transient_error_keeps_running(run_main, capsys):
    """일시적 API 오류가 나면 안내 후 대화를 계속하고, 다음 입력은 정상 처리된다.

    대상: app.main 오류 처리 (errors.describe_error)
    구분: 예외
    우선순위: 상
    사전조건: 첫 run_turn 이 openai.APIConnectionError
    절차: '책' → '다시' → 'q'
    기대결과: stderr 에 '[오류] 외부 API 서버에 연결하지 못했습니다', 계속 안내 출력,
        run_turn 2회, 정상 종료(Goodbye!!)
    """
    import httpx
    import openai

    error = openai.APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com'))
    payloads, out = run_main(['책', '다시', 'q'], errors=[error])

    assert '[오류] 외부 API 서버에 연결하지 못했습니다' in run_main.err
    assert '대화 기록은 저장되어 있습니다' in out
    assert len(payloads) == 2
    assert 'Goodbye!!' in out


def test_main_fatal_error_exits_with_code_1(run_main, capsys):
    """키 오류 같은 설정 오류는 안내 후 종료 코드 1 로 끝내고 traceback 을 보이지 않는다.

    대상: app.main 오류 처리 (fatal)
    구분: 예외
    우선순위: 상
    사전조건: 첫 run_turn 이 openai.AuthenticationError(401)
    절차: '책' → (남은 입력 'q' 는 읽지 않음)
    기대결과: SystemExit(1), stderr 에 OPENAI_API_KEY 확인 안내, 'Traceback' 미출력, run_turn 1회
    """
    import openai

    with pytest.raises(SystemExit) as exit_info:
        run_main(['책', 'q'], errors=[api_error(openai.AuthenticationError, 401)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert 'OPENAI_API_KEY' in captured.err
    assert 'Traceback' not in captured.err + captured.out


def test_main_unexpected_error_is_logged_and_continues(run_main, capsys, caplog):
    """모르는 오류(코드 버그)는 traceback 을 로그로 남기고 대화를 계속한다.

    대상: app.main 오류 처리 (분류되지 않은 예외)
    구분: 예외
    우선순위: 중
    사전조건: 첫 run_turn 이 ValueError
    절차: '책' → 'q'
    기대결과: '예기치 못한 오류가 발생했습니다: ValueError' 안내, ERROR 로그에 예외 정보, 정상 종료
    """
    _, out = run_main(['책', 'q'], errors=[ValueError('bug')])

    assert '예기치 못한 오류가 발생했습니다: ValueError' in run_main.err
    [record] = [r for r in caplog.records if r.levelname == 'ERROR']
    assert record.exc_info and record.exc_info[0] is ValueError
    assert 'Goodbye!!' in out


def test_main_ctrl_c_during_turn_returns_to_prompt(run_main):
    """답변 중 Ctrl+C 를 누르면 이번 요청만 멈추고 다음 입력을 받는다.

    대상: app.main (KeyboardInterrupt, 실행 중)
    구분: 예외
    우선순위: 중
    사전조건: 첫 run_turn 이 KeyboardInterrupt
    절차: '책' → '다시' → 'q'
    기대결과: '[중단]' 안내, run_turn 2회, 정상 종료
    """
    payloads, out = run_main(['책', '다시', 'q'], errors=[KeyboardInterrupt()])

    assert '[중단] 이번 요청을 멈췄습니다' in out
    assert len(payloads) == 2
    assert 'Goodbye!!' in out


@pytest.mark.parametrize('last', [EOFError(), KeyboardInterrupt()], ids=['eof', 'ctrl_c_at_prompt'])
def test_main_input_end_exits_cleanly(run_main, last):
    """입력 대기 중 표준입력이 끝나거나 Ctrl+C 를 누르면 오류 없이 종료한다.

    대상: app.main (EOFError / KeyboardInterrupt, 입력 대기 중)
    구분: 예외
    우선순위: 중
    사전조건: 입력 '책' 다음에 EOF 또는 Ctrl+C
    절차: main() 실행
    기대결과: 예외 없이 반환, 'Goodbye!!' 출력, run_turn 1회
    """
    payloads, out = run_main(['책', last])
    assert len(payloads) == 1
    assert 'Goodbye!!' in out


def test_main_invalid_settings_exit_with_names(monkeypatch, make_settings):
    """설정 검증에 실패하면 문제 항목 이름만 안내하고 종료한다(값은 출력하지 않음).

    대상: app.main (pydantic ValidationError)
    구분: 예외
    우선순위: 상
    사전조건: TAVILY_API_KEY 누락, SEARCH_TOP_K=0, OPENAI_API_KEY='sk-secret'
    절차: main() 실행
    기대결과: SystemExit 메시지에 TAVILY_API_KEY, SEARCH_TOP_K 포함, 'sk-secret' 미포함
    """
    from book_agent.settings import Settings

    def broken_settings():
        return Settings(openai_api_key='sk-secret', search_top_k=0, _env_file=None)

    monkeypatch.setattr(app, 'load_settings', broken_settings)

    with pytest.raises(SystemExit) as exit_info:
        app.main()

    message = str(exit_info.value.code)
    assert message.startswith('[설정 오류]')
    assert 'TAVILY_API_KEY' in message and 'SEARCH_TOP_K' in message
    assert 'sk-secret' not in message


def test_main_recovers_after_failed_turn(run_main, make_ctx, fast_retry):
    """턴 도중 오류로 멈춘 뒤 새로 입력하면 resume 이 아니라 새 메시지로 다시 시작한다.

    대상: app.waiting_for_human + 실제 그래프 재시도/복구
    구분: 예외
    우선순위: 상
    사전조건: researcher 가 연결 오류 3회(재시도 한도 소진) 후 정상, 재시도 대기 0
    절차: '책'(실패) → '다시 해 줘'(성공) → 'q'
    기대결과: researcher 총 4회 호출(1턴 3회 실패, 2턴 성공), 두 번째 payload 는 dict(새 메시지),
        두 입력이 모두 대화에 남고 마지막은 human 대기
    """
    import httpx
    import openai
    from langchain_core.messages import AIMessage

    calls = {'n': 0}

    def flaky(_):
        calls['n'] += 1
        if calls['n'] <= 3:
            raise openai.APIConnectionError(request=httpx.Request('POST', 'https://x'))
        return {'messages': [AIMessage('요약')]}

    ctx = make_ctx(
        plans=[plan('researcher'), plan('researcher'), plan('communicator')],
        replies=['보고'],
        researcher=RunnableLambda(flaky),
    )

    payloads, _ = run_main(['책', '다시 해 줘', 'q'], ctx=ctx)

    assert calls['n'] == 4
    assert isinstance(payloads[1], dict)
    with open_checkpointer(run_main.settings.checkpoint_db) as saver:
        state = build_graph(checkpointer=saver).get_state(config('book_001'))
    humans = [m.content for m in state.values['messages'] if m.type == 'human']
    assert humans == ['책', '다시 해 줘']
    assert state.next == ('human',)


def test_stdio_is_switched_to_utf8(monkeypatch):
    """cp949 로 열린 입출력도 UTF-8 로 바뀌어 em dash·이모지가 그대로 오간다.

    대상: app.use_utf8_io
    구분: 예외
    우선순위: 상
    사전조건: stdin/stdout/stderr 를 cp949 스트림으로 교체(Windows 에서 파이프로 연결한 상황),
        stdin 에는 UTF-8 로 인코딩한 한글 입력
    절차: use_utf8_io() 후 입력 한 줄 읽기, '목차 — 완료 ✅' 출력
    기대결과: 세 스트림 모두 utf-8, 입력 한글이 깨지지 않음, 출력 바이트가 UTF-8 원문과 동일
    """
    import io
    import sys

    stdin = io.TextIOWrapper(io.BytesIO('파이썬 책 — 초보자용\n'.encode()), encoding='cp949')
    out_buffer = io.BytesIO()
    stdout = io.TextIOWrapper(out_buffer, encoding='cp949', newline='\n')
    stderr = io.TextIOWrapper(io.BytesIO(), encoding='cp949')
    monkeypatch.setattr(sys, 'stdin', stdin)
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)

    app.use_utf8_io()
    line = sys.stdin.readline()
    print('목차 — 완료 ✅')
    stdout.flush()

    assert {s.encoding for s in (stdin, stdout, stderr)} == {'utf-8'}
    assert line == '파이썬 책 — 초보자용\n'
    assert out_buffer.getvalue().decode('utf-8') == '목차 — 완료 ✅\n'


def test_utf8_io_replaces_invalid_input_bytes(monkeypatch):
    """UTF-8 이 아닌 바이트가 들어와도 멈추지 않고 대체 문자로 읽는다.

    대상: app.use_utf8_io (errors='replace')
    구분: 경계
    우선순위: 하
    사전조건: stdin 에 cp949 로 인코딩한 '한글'(UTF-8 로는 잘못된 바이트)
    절차: use_utf8_io() 후 입력 한 줄 읽기
    기대결과: UnicodeDecodeError 없음, 대체 문자(U+FFFD) 포함
    """
    import io
    import sys

    stdin = io.TextIOWrapper(io.BytesIO('한글\n'.encode('cp949')), encoding='cp949')
    monkeypatch.setattr(sys, 'stdin', stdin)

    app.use_utf8_io()

    assert '�' in sys.stdin.readline()
