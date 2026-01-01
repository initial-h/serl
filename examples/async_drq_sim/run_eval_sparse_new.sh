export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sparse_sim.py "$@" \
    --eval \
    --checkpoint_path ./checkpoints/checkpoint_0 \
    --render True \
    --eval_n_trajs 10