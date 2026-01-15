debug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

# use the environment in install_dsrl.sh
TASK_NAME=('Libero-Spatial' 'Mustard-Place' 'Corn-in-Basket' 'Letter-Arrange' 'Shoe-on-Rack' 'Mug-Insert')

echo "Running test script for environment: ${TASK_NAME[0]}"
python scripts/test_env.py \
--env_id="${TASK_NAME[0]}" \
--seed=0