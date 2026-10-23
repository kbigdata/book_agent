'''연산 장치 선택. torch가 없는 PC(--no-group gpu)에서도 동작한다.'''

def resolve_device(preference:str='auto')->str:
    if preference != 'auto':
        return preference

    try:
        import torch
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'