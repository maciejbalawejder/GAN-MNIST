# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, LeakyReLU, Flatten, ReLU, Reshape, Conv2DTranspose
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
tf.test.is_gpu_available()

# DATASET
def plot(dataset):
    for i in range(25):
        plt.subplot(5,5,i + 1)
        plt.axis('off')
        plt.imshow(dataset[i],cmap = 'gray')
    plt.show()

(x_train, _ ), (_ , _) = tf.keras.datasets.mnist.load_data()
plot(x_train)
x_train = np.expand_dims(x_train,axis = -1)
x_train = x_train.astype('float32') / 255
x_train = x_train[:30000]
print("Image shape : {}, Size of the dataset: {}".format(x_train[0].shape, x_train.shape[0]))

# DISCRIMINATOR
def Discriminator(img_shape = (28,28,1)):
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same', input_shape = img_shape))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))

    optimizer = Adam(lr=0.0002, beta_1 = 0.5)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

# GENERATOR
def Generator(latent_dim):
    model = Sequential()

    n_nodes = 128*7*7
    model.add(Dense(n_nodes, input_dim = latent_dim))
    model.add(LeakyReLU(alpha = 0.2))
    # model.add(BatchNormalization(momentum = 0.9))
    model.add(Reshape((7,7,128)))

    # Upsample 7x7 feature map to 14x14
    model.add(Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    # model.add(BatchNormalization(momentum = 0.9))

    # Upsample 14x14 feature map to 28x28 image
    model.add(Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    # model.add(BatchNormalization(momentum = 0.9))
    model.add(Conv2D(filters = 1, kernel_size = (7,7), activation = "sigmoid", padding = 'same'))

    return model
# SAMPLES
def true_samples(dataset, no_sample):
    index = np.random.randint(0,dataset.shape[0],no_sample)

    images = dataset[index]

    labels = np.ones((no_sample,1))
    return images,labels

def fake_samples(generator, input_dim, no_sample):
    noise = np.random.randn(input_dim*no_sample).reshape(no_sample, input_dim)
    print(noise.shape)

    gen_images = generator.predict(noise)

    labels = np.zeros((no_sample,1))
    return gen_images,labels

# GAN
def GAN(generator, discriminator):
    discriminator.trainable = False

    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    optimizer = Adam(lr = 0.0002, beta_1 = 0.5)

    model.compile(optimizer=optimizer, loss = 'binary_crossentropy')
    return model

# TRAINING
def save(model,epoch,s=6):
    x_fake, _ = fake_samples(model,128,s*s)
    print(x_fake.shape)
    for i in range(s*s):
        plt.subplot(s,s,1+i)
        plt.axis('off')
        plt.imshow(x_fake[i])
    plt.savefig(r'GAN/epoch{}.png'.format(epoch))
    plt.close()

def summerize(g_loss, d_loss):
    epoches = np.arange(len(g_loss))
    plt.plot(g_loss, epoches)
    plt.plot(d_loss, epoches)
    plt.legend(["g_loss", "d_loss"])
    plt.save(r"GAN/Performance.png")

def train(gan, generator, discriminator,dataset, epoches = 100, batch = 64, latent_dim = 128):
    half_batch = batch//2
    for i in range(epoches):
        for j in range(half_batch):
            x_fake, y_fake = fake_samples(generator, latent_dim, half_batch)
            x_real, y_real = true_samples(dataset, half_batch)
            X,y = np.vstack((x_real,x_fake)), np.vstack((y_real,y_fake))
            _, d_loss = discriminator.train_on_batch(X,y)

            x_gan = np.random.randn(latent_dim*batch).reshape(batch, latent_dim)
            y_gan = np.ones((batch,1))
            g_loss = gan.train_on_batch(x_gan, y_gan)

        save(generator,str(i)+str(j))
        gan.save(r'GAN/model.{}.h5'.format(i))
    summerize(g_loss,d_loss)

# TEST
generator = Generator(128)
discriminator = Discriminator()
gan = GAN(generator, discriminator)
train(gan, generator, discriminator, x_train)
