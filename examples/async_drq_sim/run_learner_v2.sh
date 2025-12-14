export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_v2_sim.py \
    --learner \
    --env PandaPickCubeVision-v0 \
    --exp_name=drq_v2_test \
    --batch_size=256 \
    --max_steps=1000000 \
    --steps_per_update=1 \
    --training_starts=2000 \
    --random_steps=2000 \
    --critic_actor_ratio=1
