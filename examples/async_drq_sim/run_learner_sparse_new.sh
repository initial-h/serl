export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_sparse_sim.py "$@" \
    --learner \
    --exp_name drq_sparse_test \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path franka_lift_cube_image_20_trajs_dense_generated.pkl \
    --reward_type sparse \
    --checkpoint_path ./checkpoints \
    --checkpoint_period 10000 \
    --save_model True
