from DroneSimEnv_movingTarget import DroneSimEnv
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=300, num_hid_layers=3)

env = DroneSimEnv()

# generate model
# ----------------------------------------
ob_space = env.observation_space
ac_space = env.action_space
pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
clip_param = clip_param * lrmult # Annealed cliping parameter epislon

ob = U.get_placeholder_cached(name="ob")
ac = pi.pdtype.sample_placeholder([None])

kloldnew = oldpi.pd.kl(pi.pd)
ent = pi.pd.entropy()
meankl = tf.reduce_mean(kloldnew)
meanent = tf.reduce_mean(ent)
pol_entpen = (-entcoeff) * meanent

ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
surr1 = ratio * atarg # surrogate from conservative policy iteration
surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
total_loss = pol_surr + pol_entpen + vf_loss
losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

var_list = pi.get_trainable_variables()
lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
adam = MpiAdam(var_list, epsilon=adam_epsilon)

assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

U.initialize()
adam.sync()
# ----------------------------------------

# configuration
timesteps_per_actorbatch = 2048
stochastic = True
max_iteration = 200

# session
with U.single_threaded_session() as sess:
    agent.initialize(sess)

    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), '/home/projectvenom/Documents/AIPilot/AIPilot-ProjectVenom-master/model_moving/exp300/Exp4_mv_best') 

    iteration = 0

    while iteration < max_iteration:
    	iteration += 1

    	ob = env.reset()

    	new = False
    	while not new:
    		ac, vpred = pi.act(stochastic, ob)
    		ob, rew, new, info = env.step(ac)

    		if new:
    			print('distance: ', info['distance'])

    env.stop()