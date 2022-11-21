echo "Start Testing......"

# echo "RL eval"
# python3 RL_eval.py --RL_policy DQN --load_model default

echo "RL_stream_Controller"
python3 RL_stream_controller.py --RL_policy DQN --data_file data_0 --load_model default

echo "End Testing......"