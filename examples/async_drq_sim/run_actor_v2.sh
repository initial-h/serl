export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_v2_sim.py \
    --actor \
    --env PandaPickCubeVision-v0 \
    --exp_name=drq_v2_test \
    --max_steps=1000000 \
    --steps_per_update=1 \
    --random_steps=2000 \
    --critic_actor_ratio=1
