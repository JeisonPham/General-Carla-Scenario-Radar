# << Leaderboard setting
# ===> pls remeber to change this one
export CODE_FOLDER=/home/jason/Desktop/mmfn
export CARLA_ROOT=/home/jason/CARLA_0.9.10.1
export HYDRA_FULL_ERROR=1
# ===> pls remeber to change this one
export SCENARIO_RUNNER_ROOT=${CODE_FOLDER}/scenario_runner
export LEADERBOARD_ROOT=${CODE_FOLDER}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":"${CODE_FOLDER}/team_code":${PYTHONPATH}

echo PYTHONPATH
#python run_steps/phase0_run_eval.py
