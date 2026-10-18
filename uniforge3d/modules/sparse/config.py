from typing import Literal

CONV = 'flex_gemm' 
DEBUG = False
ATTN = 'flash_attn'
_ATTN_BACKENDS = ('xformers', 'flash_attn', 'flash_attn_3', 'sdpa')
# A conservative attention score/workspace estimate, NOT a total RSS limit.
SDPA_SCORE_WORKSPACE_BYTES = 512 * 1024 * 1024
SDPA_QUERY_CHUNK_SIZE = 256

def __from_env():
    import os
    
    global CONV
    global DEBUG
    global ATTN
    global SDPA_SCORE_WORKSPACE_BYTES
    global SDPA_QUERY_CHUNK_SIZE
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend not in (None, 'flex_gemm'):
        raise ValueError(
            'This RecGen inference package only includes SPARSE_CONV_BACKEND=flex_gemm; '
            f'got {env_sparse_conv_backend!r}'
        )
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None:
        if env_sparse_attn_backend not in _ATTN_BACKENDS:
            raise ValueError(f'Unknown sparse attention backend: {env_sparse_attn_backend!r}')
        ATTN = env_sparse_attn_backend
    for name in ('SDPA_SCORE_WORKSPACE_BYTES', 'SDPA_QUERY_CHUNK_SIZE'):
        value = os.environ.get('SPARSE_' + name)
        if value is not None:
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ValueError(f'SPARSE_{name} must be a positive integer') from exc
            if parsed <= 0:
                raise ValueError(f'SPARSE_{name} must be a positive integer')
            globals()[name] = parsed


__from_env()
    

def set_conv_backend(backend: Literal['flex_gemm']):
    global CONV
    if backend != 'flex_gemm':
        raise ValueError('RecGen only includes the flex_gemm sparse-convolution backend')
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']):
    global ATTN
    if backend not in _ATTN_BACKENDS:
        raise ValueError(f'Unknown sparse attention backend: {backend!r}')
    ATTN = backend
