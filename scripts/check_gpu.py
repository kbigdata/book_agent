"""
GPU 개발환경 점검 스크립트.

    uv run python scripts/check_gpu.py            # PyTorch / CUDA 점검
    uv run python scripts/check_gpu.py --embed    # + 로컬 임베딩 모델(GPU) 실행 점검
"""

import argparse
import platform
import sys
import time


def check_torch() -> bool:
    import torch

    print(f'Python        : {sys.version.split()[0]} ({platform.system()})')
    print(f'PyTorch       : {torch.__version__} (CUDA 빌드 {torch.version.cuda})')
    if not torch.cuda.is_available():
        print('CUDA 사용 불가 → 드라이버 / torch 인덱스(cu130) 설정을 확인하세요.')
        return False

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    memory = props.total_memory / 1024**3
    print(f'GPU           : {props.name} (sm_{major}{minor}, {memory:.1f} GiB)')
    print(f'지원 아키텍처 : {", ".join(torch.cuda.get_arch_list())}')

    # 카드마다 FP16 이 FP32 보다 느릴 수도 있으므로 둘 다 재 본다.
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(4096, 4096, device='cuda', dtype=dtype)
        x @ x  # 워밍업: 첫 호출의 cuBLAS 초기화 시간은 측정에서 뺀다.
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()
        tflops = 10 * 2 * 4096**3 / (time.perf_counter() - start) / 1e12
        print(f'{dtype!s:<14}: {tflops:.2f} TFLOPS (4096x4096 행렬곱)')
    return True


def check_embeddings(model_name:str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    start=time.perf_counter()
    model=SentenceTransformer(model_name, device='cuda')
    print(f'모델 로딩 : {model_name} ({time.perf_counter() - start:.1f}s)')

    texts=['개발환경 점검 문장입니다.','GPU embedding smoke test'] * 32

    start=time.perf_counter()
    vectors=model.encode(texts, normalize_embeddings=True)
    print(
        f'임베딩      : {vectors.shape[0]}개 X {vectors.shape[1]} 차원 '
        f'{time.perf_counter() - start:.2f}s'
    )
    print(f'GPU 메모리 : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB (최대 사용)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', action='store_true', help='임베딩 모델도 점검')
    parser.add_argument('--model', default='BAAI/bge-m3', help='점검할 임베딩 모델')
    args = parser.parse_args()

    if check_torch() and args.embed:
        check_embeddings(args.model)
