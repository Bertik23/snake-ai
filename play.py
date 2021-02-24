from snakeAI import Agent
from tensorflow import keras

agent = Agent()

agent.actor = keras.models.load_model("models\\")
