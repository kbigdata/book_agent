"""
테스트 시나리오 시트(엑셀) 생성 스크립트.

테스트 docstring 의 시나리오(대상/구분/우선순위/사전조건/절차/기대결과)와
pytest 실행 결과를 합쳐 한 장의 보고서로 만든다.
결과 JSON 을 여러 개 주면 하나로 합친다(같은 테스트는 뒤 파일의 결과를 쓰되, SKIP 은 무시).
단위 테스트와 비용이 드는 live/통합 테스트를 따로 실행해도 한 보고서로 묶을 수 있다.

사용법 (프로젝트 루트에서):
    uv run pytest --live --scenario-report=docs/unit_test_results.json
    uv run --with openpyxl python scripts/build_test_report.py \
        docs/unit_test_report.xlsx docs/unit_test_results.json [추가 결과.json ...]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

# 테스트 파일 -> (모듈 코드, 대상 모듈)
MODULES = {
    'test_environment.py': ('ENV', '실행 환경 (Python, .env, CUDA)'),
    'test_settings.py': ('SET', 'settings.py / device.py'),
    'test_models_state.py': ('MOD', 'models.py / state.py'),
    'test_repositories.py': ('REP', 'repositories.py'),
    'test_errors.py': ('ERR', 'errors.py'),
    'test_services.py': ('SVC', 'services.py'),
    'test_tools.py': ('TOL', 'tools.py'),
    'test_nodes.py': ('NOD', 'nodes.py'),
    'test_graph.py': ('GRP', 'graph.py'),
    'test_app.py': ('APP', 'app.py / __init__.py'),
    'test_live.py': ('LIV', '실제 API 통합 (OpenAI, Tavily)'),
    'test_integration.py': ('INT', 'CLI 통합 (book-agent 프로세스)'),
}
OUTCOMES = {'passed': 'PASS', 'failed': 'FAIL', 'error': 'ERROR', 'skipped': 'SKIP'}
FIELDS = ('대상', '구분', '우선순위', '사전조건', '절차', '기대결과')

FONT = 'Arial'
HEADER_FILL = PatternFill('solid', fgColor='1F3864')
SUB_FILL = PatternFill('solid', fgColor='D9E1F2')
THIN = Side(style='thin', color='BFBFBF')
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical='top')
CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)

# 시나리오 시트 열 정의: (머리말, 너비)
COLUMNS = [
    ('ID', 9),
    ('모듈', 7),
    ('테스트 파일', 18),
    ('테스트 대상', 26),
    ('구분', 7),
    ('우선순위', 8),
    ('시나리오', 40),
    ('사전조건', 34),
    ('절차', 34),
    ('기대결과', 44),
    ('결과', 8),
    ('소요(초)', 9),
    ('비고(실패/스킵 사유)', 30),
    ('테스트 함수', 40),
]
COL = {name: get_column_letter(i) for i, (name, _) in enumerate(COLUMNS, 1)}
SHEET = '시나리오'


def parse_doc(doc: str) -> dict[str, str]:
    """docstring 을 제목과 'key: value' 항목으로 나눈다. 들여쓴 이어지는 줄은 앞 항목에 붙인다."""
    title, _, body = doc.partition('\n\n')
    result = {'시나리오': ' '.join(title.split())}
    current = None
    for line in body.splitlines():
        match = re.match(rf'^({"|".join(FIELDS)}):\s*(.*)$', line)
        if match:
            current = match.group(1)
            result[current] = match.group(2).strip()
        elif current and line.strip():
            result[current] += ' ' + line.strip()
    return result


def build_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows, seq = [], Counter()
    # 하위 모듈(설정/모델)부터 상위 모듈(그래프/앱) 순으로 정렬한다. 파일 안에서는 실행 순서 유지.
    order = list(MODULES)
    results = sorted(results, key=lambda r: order.index(r['file']) if r['file'] in order else 99)
    for r in results:
        code, _ = MODULES.get(r['file'], ('ETC', r['file']))
        seq[code] += 1
        info = parse_doc(r['doc'])
        title = info['시나리오'] + (f' [{r["param"]}]' if r['param'] else '')
        message = r['message'] if r['outcome'] != 'passed' else ''
        rows.append(
            {
                'ID': f'{code}-{seq[code]:02d}',
                '모듈': code,
                '테스트 파일': r['file'],
                '테스트 대상': info.get('대상', ''),
                '구분': info.get('구분', ''),
                '우선순위': info.get('우선순위', ''),
                '시나리오': title,
                '사전조건': info.get('사전조건', ''),
                '절차': info.get('절차', ''),
                '기대결과': info.get('기대결과', ''),
                '결과': OUTCOMES[r['outcome']],
                '소요(초)': round(r['duration'], 3),
                '비고(실패/스킵 사유)': message.strip()[-500:],
                '테스트 함수': r['nodeid'],
            }
        )
    return rows


def style_header(ws: Worksheet, row: int, first: int, last: int) -> None:
    for c in range(first, last + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color='FFFFFF', size=10)
        cell.fill = HEADER_FILL
        cell.alignment = CENTER
        cell.border = BORDER


def add_result_colors(ws: Worksheet, ref: str) -> None:
    """결과 값에 따라 색을 칠한다(조건부 서식이라 값을 고치면 색도 바뀐다)."""
    colors = {
        'PASS': ('C6EFCE', '006100'),
        'FAIL': ('FFC7CE', '9C0006'),
        'ERROR': ('FCE4D6', 'C65911'),
        'SKIP': ('EDEDED', '595959'),
    }
    for value, (fill, font) in colors.items():
        ws.conditional_formatting.add(
            ref,
            CellIsRule(
                operator='equal',
                formula=[f'"{value}"'],
                fill=PatternFill('solid', fgColor=fill, bgColor=fill),
                font=Font(name=FONT, color=font, bold=True),
            ),
        )


def write_scenarios(wb: Workbook, rows: list[dict[str, Any]]) -> int:
    ws = wb.create_sheet(SHEET)
    for i, (name, width) in enumerate(COLUMNS, 1):
        ws.cell(row=1, column=i, value=name)
        ws.column_dimensions[get_column_letter(i)].width = width
    style_header(ws, 1, 1, len(COLUMNS))

    centered = {'ID', '모듈', '구분', '우선순위', '결과', '소요(초)'}
    for r, row in enumerate(rows, 2):
        for c, (name, _) in enumerate(COLUMNS, 1):
            cell = ws.cell(row=r, column=c, value=row[name])
            cell.font = Font(name=FONT, size=9)
            cell.border = BORDER
            cell.alignment = CENTER if name in centered else WRAP_TOP
            if name == '소요(초)':
                cell.number_format = '0.000'

    last = len(rows) + 1
    ws.freeze_panes = 'H2'
    ws.auto_filter.ref = f'A1:{get_column_letter(len(COLUMNS))}{last}'
    add_result_colors(ws, f'{COL["결과"]}2:{COL["결과"]}{last}')
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = '1:1'
    return last


def rng(name: str, last: int) -> str:
    return f'{SHEET}!${COL[name]}$2:${COL[name]}${last}'


def write_summary(ws: Worksheet, payload: dict[str, Any], last: int) -> None:
    result, module, kind, prio, dur = (
        rng('결과', last),
        rng('모듈', last),
        rng('구분', last),
        rng('우선순위', last),
        rng('소요(초)', last),
    )
    ws['A1'] = 'book-agent 테스트 결과 보고서 (단위 · 실제 API · CLI 통합)'
    ws['A1'].font = Font(name=FONT, bold=True, size=16, color='1F3864')
    ws['A2'] = (
        '테스트 시나리오와 실행 결과. 아래 집계는 모두 '
        f"'{SHEET}' 시트를 참조하는 수식이라 결과 값을 고치면 자동으로 다시 계산된다."
    )
    ws['A2'].font = Font(name=FONT, size=9, italic=True, color='595959')

    info = [
        ('실행 시작', payload['started_at'].replace('T', ' ')),
        ('실행 종료', payload['finished_at'].replace('T', ' ')),
        ('실행 명령', payload['command']),
        ('pytest 종료 코드', f'{payload["exit_status"]} (0 = 전체 통과)'),
        ('총 소요 시간(초)', f'=SUM({dur})'),
    ]
    for i, (key, value) in enumerate(info, 4):
        ws.cell(row=i, column=1, value=key).font = Font(name=FONT, bold=True, size=10)
        cell = ws.cell(row=i, column=2, value=value)
        cell.font = Font(name=FONT, size=10)
        cell.alignment = Alignment(horizontal='left')
    ws['B8'].number_format = '0.00'

    # ---- 전체 집계
    ws['A10'] = '전체 결과'
    ws['A10'].font = Font(name=FONT, bold=True, size=12, color='1F3864')
    heads = ['전체', 'PASS', 'FAIL', 'ERROR', 'SKIP', '통과율']
    for i, h in enumerate(heads, 1):
        ws.cell(row=11, column=i, value=h)
    style_header(ws, 11, 1, len(heads))
    totals = [
        f'=COUNTA({result})',
        f'=COUNTIF({result},"PASS")',
        f'=COUNTIF({result},"FAIL")',
        f'=COUNTIF({result},"ERROR")',
        f'=COUNTIF({result},"SKIP")',
        '=IF(A12-E12=0,"-",B12/(A12-E12))',
    ]
    for i, f in enumerate(totals, 1):
        cell = ws.cell(row=12, column=i, value=f)
        cell.font = Font(name=FONT, bold=True, size=12)
        cell.alignment = CENTER
        cell.border = BORDER
    ws['F12'].number_format = '0.0%'
    ws['A13'] = (
        '통과율 = PASS / (전체 - SKIP). SKIP 은 실행 조건(GPU 등)이 맞지 않아 건너뛴 시나리오.'
    )
    ws['A13'].font = Font(name=FONT, size=9, italic=True, color='595959')

    # ---- 모듈별 집계
    start = 15
    ws.cell(row=start, column=1, value='모듈별 결과').font = Font(
        name=FONT, bold=True, size=12, color='1F3864'
    )
    heads = ['모듈', '대상', '시나리오 수', 'PASS', 'FAIL', 'ERROR', 'SKIP', '통과율', '소요(초)']
    for i, h in enumerate(heads, 1):
        ws.cell(row=start + 1, column=i, value=h)
    style_header(ws, start + 1, 1, len(heads))
    r = start + 2
    first = r
    for code, target in MODULES.values():
        values = [
            code,
            target,
            f'=COUNTIF({module},A{r})',
            f'=COUNTIFS({module},A{r},{result},"PASS")',
            f'=COUNTIFS({module},A{r},{result},"FAIL")',
            f'=COUNTIFS({module},A{r},{result},"ERROR")',
            f'=COUNTIFS({module},A{r},{result},"SKIP")',
            f'=IF(C{r}-G{r}=0,"-",D{r}/(C{r}-G{r}))',
            f'=SUMIF({module},A{r},{dur})',
        ]
        write_row(ws, r, values)
        r += 1
    write_row(
        ws,
        r,
        ['합계', '']
        + [f'=SUM({c}{first}:{c}{r - 1})' for c in 'CDEFG']
        + [f'=IF(C{r}-G{r}=0,"-",D{r}/(C{r}-G{r}))', f'=SUM(I{first}:I{r - 1})'],
        bold=True,
    )
    for row in range(first, r + 1):
        ws[f'H{row}'].number_format = '0.0%'
        ws[f'I{row}'].number_format = '0.00'

    # ---- 구분별 / 우선순위별
    r += 2
    for title, column, keys in (
        ('구분별 결과', kind, ['정상', '예외', '경계']),
        ('우선순위별 결과', prio, ['상', '중', '하']),
    ):
        ws.cell(row=r, column=1, value=title).font = Font(
            name=FONT, bold=True, size=12, color='1F3864'
        )
        heads = ['항목', '', '시나리오 수', 'PASS', 'FAIL', 'ERROR', 'SKIP', '통과율']
        for i, h in enumerate(heads, 1):
            ws.cell(row=r + 1, column=i, value=h)
        style_header(ws, r + 1, 1, len(heads))
        r += 2
        for key in keys:
            values = [
                key,
                '',
                f'=COUNTIF({column},A{r})',
                f'=COUNTIFS({column},A{r},{result},"PASS")',
                f'=COUNTIFS({column},A{r},{result},"FAIL")',
                f'=COUNTIFS({column},A{r},{result},"ERROR")',
                f'=COUNTIFS({column},A{r},{result},"SKIP")',
                f'=IF(C{r}-G{r}=0,"-",D{r}/(C{r}-G{r}))',
            ]
            write_row(ws, r, values)
            ws[f'H{r}'].number_format = '0.0%'
            r += 1
        r += 1

    ws.cell(row=r, column=1, value='구분 정의').font = Font(name=FONT, bold=True, size=10)
    legend = [
        '정상: 기대한 입력에서 기능이 올바르게 동작하는지',
        '예외: 잘못된 입력·외부 오류에서 안전하게 실패/복구하는지',
        '경계: 한도·빈 값·중복 등 경계 조건에서 동작하는지',
        '우선순위 상: 핵심 흐름(라우팅, 저장, 복원) / 중: 보조 기능 / 하: 표시·편의 기능',
    ]
    for i, text in enumerate(legend, r + 1):
        ws.cell(row=i, column=1, value=text).font = Font(name=FONT, size=9, color='595959')

    for col, width in zip('ABCDEFGHI', (16, 30, 12, 9, 9, 9, 9, 10, 10), strict=True):
        ws.column_dimensions[col].width = width


def write_row(ws: Worksheet, r: int, values: list[Any], bold: bool = False) -> None:
    for c, value in enumerate(values, 1):
        cell = ws.cell(row=r, column=c, value=value)
        cell.font = Font(name=FONT, size=10, bold=bold)
        cell.border = BORDER
        cell.alignment = CENTER if c != 2 else Alignment(vertical='center')
        if bold:
            cell.fill = SUB_FILL


def write_environment(wb: Workbook, payload: dict[str, Any]) -> None:
    ws = wb.create_sheet('실행 환경')
    ws.append(['항목', '값'])
    style_header(ws, 1, 1, 2)
    rows = [
        ('Python', payload['python']),
        ('플랫폼', payload['platform']),
        ('연산 장치', payload['device']),
        *((f'패키지: {k}', v) for k, v in payload['packages'].items()),
        ('외부 의존성', 'OpenAI/Tavily API 는 호출하지 않음(가짜 LLM·검색 클라이언트·임베딩 사용)'),
        ('파일 시스템', '모든 파일은 pytest tmp_path 아래에만 생성'),
        ('결과 값', 'PASS 통과 / FAIL 단언 실패 / ERROR 준비·정리 단계 오류 / SKIP 조건 미충족'),
        ('ID 규칙', '모듈 코드-일련번호. 매개변수화 테스트는 케이스마다 ID 를 하나씩 부여'),
    ]
    for key, value in rows:
        ws.append([key, value])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name=FONT, size=10, bold=cell.column == 1)
            cell.border = BORDER
            cell.alignment = WRAP_TOP
    ws.column_dimensions['A'].width = 24
    ws.column_dimensions['B'].width = 90


def merge(sources: list[Path]) -> dict[str, Any]:
    """여러 실행 결과를 합친다. 실행 정보는 가장 이른 시작~가장 늦은 종료로 묶는다."""
    payloads = [json.loads(src.read_text(encoding='utf-8')) for src in sources]
    merged = dict(payloads[-1])
    merged['started_at'] = min(p['started_at'] for p in payloads)
    merged['finished_at'] = max(p['finished_at'] for p in payloads)
    merged['exit_status'] = max(p['exit_status'] for p in payloads)
    merged['command'] = ' ; '.join(p['command'] for p in payloads)
    results: dict[str, dict[str, Any]] = {}
    for p in payloads:
        for r in p['results']:
            previous = results.get(r['nodeid'])
            # --live 없이 돌린 실행의 SKIP 이 앞선 실제 실행 결과를 덮어쓰지 않게 한다.
            if previous and r['outcome'] == 'skipped' and previous['outcome'] != 'skipped':
                continue
            results[r['nodeid']] = r
    merged['results'] = list(results.values())
    return merged


def main(dst: Path, sources: list[Path]) -> None:
    payload = merge(sources)
    rows = build_rows(payload['results'])

    wb = Workbook()
    summary = wb.active
    summary.title = '요약'
    last = write_scenarios(wb, rows)
    write_summary(summary, payload, last)
    write_environment(wb, payload)
    wb.calculation.fullCalcOnLoad = True  # 열 때 수식을 다시 계산한다

    dst.parent.mkdir(parents=True, exist_ok=True)
    wb.save(dst)

    counts = Counter(r['결과'] for r in rows)
    print(f'{dst}: 시나리오 {len(rows)}건, ' + ', '.join(f'{k} {v}' for k, v in counts.items()))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('사용법: build_test_report.py <보고서.xlsx> <결과.json> [결과.json ...]')
    main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])
