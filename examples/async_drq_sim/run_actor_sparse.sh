export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim.py "$@" \
    --actor \
    --exp_name drq_sparse_test \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --reward_type sparse \
    --render
