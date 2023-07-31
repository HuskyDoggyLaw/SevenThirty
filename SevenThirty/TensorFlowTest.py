
import tensorflow as tf
from tensorflow.keras import layers

# Define the generator model
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(784, activation='tanh'))  # 784 = 28*28 (MNIST image size)
    model.add(layers.Reshape((28, 28, 1)))  # Reshape to 28x28 grayscale image
    return model

# Create the generator
latent_dim = 100  # Size of the generator's input noise vector
generator = build_generator(latent_dim)

# Generate a fake image
noise = tf.random.normal(shape=(1, latent_dim))
fake_image = generator.predict(noise)

# Display the fake image
import matplotlib.pyplot as plt
plt.imshow(fake_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()
