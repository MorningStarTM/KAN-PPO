env_name = "carRace"
has_continuous_action_space = False

max_ep_len = 900                    # max timesteps in one episode
max_training_timesteps = int(5e4)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = None

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.00003       # learning rate for actor network
lr_critic = 0.0001       # learning rate for critic network

random_seed = 0    