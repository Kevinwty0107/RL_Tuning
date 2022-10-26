echo "Start Testing......"

# echo "Default Seeting"
# python3 controller.py --search_method default --data_file data_0
# wait
# echo "Random Search"
# python3 controller.py --search_method random_search --data_file data_0
# wait

echo "BO"
python3 controller.py --search_method BO --data_file data_0
wait

# echo "grid_search"
# python3 controller.py --search_method grid_search --data_file data_0
# wait

echo "RL"
python3 RL_controller.py --RL_policy DQN --data_file data_0 --save_model True

echo "End Testing......"


