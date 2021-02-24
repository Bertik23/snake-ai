import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard

from hra import Hra


def getAdvantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i+1 == len(rewards):
            continue
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lambda_ * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss(oldpolicyProbs, advantages, rewards, values):
    def loss(yTrue, yPred):
        print(oldpolicyProbs, advantages, rewards, values)
        newpolicyProbs = yPred
        ratio = K.exp(
            K.log(newpolicyProbs + 1e-10) - K.log(oldpolicyProbs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(
            ratio,
            min_value=1 - clippingVal,
            max_value=1 + clippingVal) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = criticDiscount * critic_loss
        + actor_loss - entropyBeta * K.mean(
            - (newpolicyProbs * K.log(newpolicyProbs + 1e-10)))
        return total_loss

    return loss


def testReward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = actorNetwork.predict(
            [state_input,
             dummy_n,
             dummy_1,
             dummy_1,
             dummy_1],
            steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward


layersList = []

layersList.append(layers.Input(1602, 128, name="input"))
layersList.append(layers.Dense(512, activation="relu")(layersList[-1]))
layersList.append(layers.Dense(512, activation="relu")(layersList[-1]))
layersList.append(layers.Dense(1, activation="tanh")(layersList[-1]))


criticNetwork = models.Model(inputs=[layersList[0]], outputs=[layersList[-1]])
criticNetwork.compile(optimizer=optimizers.Adam(), loss="mse")

layersList = []

layersList.append(layers.Input(1602, 128, name="input"))
layersList.append(layers.Dense(512, activation="relu")(layersList[-1]))
layersList.append(layers.Dense(512, activation="relu")(layersList[-1]))
layersList.append(layers.Dense(4, activation="softmax")(layersList[-1]))
# nnoldpolicyProbs = layers.Input(shape=(1, 4,))
# nnadvantages = layers.Input(shape=(1, 1,))
# nnrewards = layers.Input(shape=(1, 1,))
# nnvalues = layers.Input(shape=(1, 1,))
nnoldpolicyProbs = layers.Input(4, 128)
nnadvantages = layers.Input(1, 128)
nnrewards = layers.Input(1, 128)
nnvalues = layers.Input(1, 128)


actorNetwork = models.Model(
    inputs=[layersList[0],
            nnoldpolicyProbs,
            nnadvantages,
            nnrewards,
            nnvalues],
    outputs=[layersList[-1]])
actorNetwork.compile(optimizer=optimizers.Adam(), loss=ppo_loss(
    nnoldpolicyProbs,
    nnadvantages,
    nnrewards,
    nnvalues))


env = Hra()
n_actions = 4


tensorBoardLog = TensorBoard(log_dir="./logs")

dummy_n = np.zeros((1, n_actions))
dummy_1 = K.expand_dims([np.zeros((1, 1, 1))], 0)
print(dummy_n)

# SETTINGS
ppoSteps = 128
gamma = .99
lambda_ = .95
clippingVal = .2
criticDiscount = .5
entropyBeta = .001
epochs = 10
# END SETTINGS

states = []
actions = []
values = []
masks = []
rewards = []
actionsProbs = []
actionsOnehot = []
advantages = []

for epoch in range(epochs):
    state = env.reset()

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actionsProbs = []
    actionsOnehot = []
    for itr in range(ppoSteps+1):
        stateInput = K.expand_dims(state, 0)
        # print(len(stateInput), len(dummy_n), len(dummy_1))
        # print(stateInput, dummy_n.shape)
        actionDist = actorNetwork.predict(
            [stateInput,
             dummy_n,
             dummy_1,
             dummy_1,
             dummy_1],
            steps=1)
        qValue = criticNetwork.predict([stateInput], steps=1)[0][0]
        action = np.random.choice(n_actions, p=actionDist[0, :])
        actionOnehot = np.zeros(n_actions)
        actionOnehot[action] = 1

        observation, reward, done, info = env.step(action)
        mask = not done

        states.append(state)
        actions.append(action)
        actionsOnehot.append(actionOnehot)
        values.append(qValue)
        masks.append(mask)
        rewards.append(reward)
        actionsProbs.append(actionDist[0])

    returns, advantages = getAdvantages(values, masks, rewards)

    print(len(states[:-1]),
          len(actionsProbs[:-1]),
          len(advantages),
          len(rewards[:-1]),
          len(values[:-1]))
    actor_loss = actorNetwork.fit(
            # x=[states, actionsProbs, advantages,
            #    np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
            # x=[states[:-1],
            #    actionsProbs[:-1],
            #    advantages,
            #    rewards[:-1],
            #    values[:-1]
            #    ],
            x=[states[:-1]],
            y=[(np.reshape(actionsOnehot, newshape=(-1, n_actions)))],
            batch_size=128,
            verbose=True, shuffle=True, epochs=8,
            callbacks=[tensorBoardLog])
    critic_loss = criticNetwork.fit([states],
                                    [np.reshape(returns, newshape=(-1, 1))],
                                    shuffle=True,
                                    epochs=8,
                                    verbose=True,
                                    callbacks=[tensorBoardLog])

    avgReward = np.mean([testReward() for _ in range(5)])
    print(f"AL = {actor_loss}, CL = {critic_loss}, avgReward = {avgReward}")


# print(states,
#       actions,
#       values,
#       masks,
#       rewards,
#       actionsProbs,
#       actionsOnehot)
