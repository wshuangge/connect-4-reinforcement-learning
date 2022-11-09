import gym
import sys



from gym_connect_four import RandomPlayer, ConnectFourEnv, SavedPlayer
from stable_baselines3 import DQN
import torch as th



env = gym.make("ConnectFour-v0")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0005, exploration_fraction=0.1, exploration_final_eps=0.1, exploration_initial_eps=1.0)
        #model = DQN.load("./model/against_random_v4")
        model.set_env(env)
        model.learn(total_timesteps=2000000, log_interval= 4)
        model.save("./model/against_minimax")
        del model
        
    elif sys.argv[1] == "inference":
        count = 0
        wins = 0
        draw = 0
        invalid = 0
        
        model = DQN.load("./model/against_minimax")
        obs = env.reset()
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render('console')
            if done:
                obs = env.reset()
                if reward == -10:
                    invalid += 1
                elif reward == 1:
                    wins += 1
                elif reward == 0.5:
                    draw += 1
                count += 1
                print(wins/count, draw/count, invalid/count)
                input()

    elif sys.argv[1] == "transfer":
        model = DQN.load("./model/against_random_v4")
        model.policy.to("cpu")
        onnxable_model = model.policy
        observation_size = model.observation_space.shape
        dummy_input = th.zeros(observation_size)
        onnx_path = "dqn_model.onnx"
        th.onnx.export(onnxable_model, dummy_input, onnx_path, opset_version=9)
                
                    
            

