"""Serial CPU-only operator tests; no model weights or native extensions."""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

import torch

torch.set_num_threads(1)
