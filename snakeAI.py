import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
import numpy as np
# import gym

from hra import Hra


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation="softmax")

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v


class Actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.a = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, input_data):
        # print(input_data)
        x = self.d1(input_data)
        a = self.a(x)
        print(a, input_data)
        return a


class Agent():
    def __init__(self):
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = Actor()
        self.critic = Critic()
        self.clip_pram = 0.2

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def learn(self, states, actions,  adv, old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p), 4))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(
            zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(
            tf.math.multiply(probability, tf.math.log(probability))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []

        for pb, t, op in zip(probability, adv, old_probs):
            t = tf.constant(t)
            op = tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math
            # .log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb, op)
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(
                ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        # print(tf.reduce_mean(tf.math.minimum(
        # sr1, sr2)) - closs + 0.001 * entropy)
        # print(tf.math.minimum(sr1, sr2) - closs + 0.001 * entropy)
        # print(sr1, sr2, closs, entropy)
        loss = tf.math.negative(tf.reduce_mean(
            tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # print(loss)
        return loss


def test_reward(env):
    total_reward = 0
    state = env.reset()
    done = False
    stepNum = 0
    while not done and stepNum < 150:
        stepNum += 1
        action = np.argmax(agentoo7.actor(np.array([state])).numpy())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if RENDER:
            env.render()

    return total_reward


def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
        g = delta + gamma * lmbda * dones[i] * g
        returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv


tf.random.set_seed(336699)
agentoo7 = Agent()
steps = 50
ep_reward = []
total_avgr = []
target = False
best_reward = 0
avg_rewards_list = []

RENDER = False

env = Hra()  # gym.make("CartPole-v0")


for s in range(steps):
    if target:
        break

    done = False
    state = env.reset()
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    values = []
    print(state)
    print("new episod")

    for e in range(128):
        action = agentoo7.act(state)
        value = agentoo7.critic(np.array([state])).numpy()
        next_state, reward, done, _ = env.step(action)
        dones.append(1-done)
        rewards.append(reward)
        states.append(state)
        # actions.append(tf.one_hot(action, 2,
        # dtype=tf.int32).numpy().tolist())
        actions.append(action)
        prob = agentoo7.actor(np.array([state]))
        probs.append(prob[0])
        values.append(value[0][0])
        state = next_state
        if RENDER:
            env.render()
        if done:
            env.reset()

    value = agentoo7.critic(np.array([state])).numpy()
    values.append(value[0][0])
    # print(probs)
    np.reshape(probs, (len(probs), 4))
    probs = np.stack(probs, axis=0)

    states, actions, returns, adv = preprocess1(
        states, actions, rewards, dones, values, 1)

    for epocs in range(10):
        al, cl = agentoo7.learn(states, actions, adv, probs, returns)
        print(f"al {al}")
        print(f"cl {cl}")
        if tf.math.is_nan(al):
            raise ValueError("al is nan")

    avg_reward = np.mean([test_reward(env) for _ in range(5)])
    print(f"total test reward is {avg_reward}")
    avg_rewards_list.append(avg_reward)
    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))
        agentoo7.actor.save(__file__.rsplit("\\", 1)[
                            0]+'\\models\\model_actor_{}_{}'.format(
                                s, avg_reward),
                            save_format="tf")
        agentoo7.critic.save(__file__.rsplit("\\", 1)[
                             0]+'\\models\\model_critic_{}_{}'.format(
                                 s, avg_reward), save_format="tf")
        best_reward = avg_reward
    # if best_reward == 200:
    #         target = True
    env.reset()

agentoo7.actor.save(__file__.rsplit("\\", 1)[
                    0]+'\\models\\model_actor_final', save_format="tf")
agentoo7.critic.save(__file__.rsplit("\\", 1)[
                     0]+'\\models\\model_critic_final', save_format="tf")
env.close()
