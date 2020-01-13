#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# In[23]:


mnist_dataset,mnist_info = tfds.load(name='mnist',with_info =True,as_supervised=True)
mnist_test,mnist_train = mnist_dataset['test'],mnist_dataset['train']


# In[24]:


num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples,tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)


def scale(image,label):
    image = tf.cast(image,tf.float32)
    image /= 255
    
    return image,label

scaled_train_and_validation_data = mnist_train.map(scale)
scaled_test_data = mnist_test.map(scale)

# hyper parameter
BUFFER_SIZE = 10000

shuffeled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffeled_train_and_validation_data.take(num_validation_samples)
training_data = shuffeled_train_and_validation_data.skip(num_validation_samples)

# hyper parameter
BATCH_SIZE = 100

training_data = training_data.batch(BATCH_SIZE)
test_data = scaled_test_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)

validation_inputs,validation_targets = next(iter(validation_data))


# In[29]:


## outlining the model
input_size = 784
output_size = 10
hidden_layer_size = 500
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(hidden_layer_size,activation ='relu'),
    tf.keras.layers.Dense(hidden_layer_size,activation ='relu'),
    tf.keras.layers.Dense(output_size,activation ='softmax')

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')


# In[30]:


NUM_EPOCHS = 5
model.fit(training_data,epochs = NUM_EPOCHS,validation_data=(validation_inputs,validation_targets),verbose=2,validation_steps=10)


# In[32]:


test_loss = model.evaluate(test_data)


# In[33]:


print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# In[ ]:




