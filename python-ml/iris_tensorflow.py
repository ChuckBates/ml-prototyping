from __future__ import absolute_import, division, print_function
from tensorflow import contrib

import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Eager execution allows immediate operation evaluation
tf.enable_eager_execution()

# Retrieve training dataset
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

# Retrieve test dataset
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

# Global definitions
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
features_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
batch_size = 32;

# Convert csv training data to tensorflow dataset
train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)

# Function: Packs features into a single array
def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# Pack features of each (features,label) pair into the training dataset
train_dataset = train_dataset.map(pack_features_vector)

# Create neural net model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# Function: measures how off a model's predictions are from the desired label
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# Function: calculate the gradients used to optimize the model
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Used to apply the gradients to the model's variables to minimize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Training loop definitions
global_step = tf.Variable(0)
tfe = contrib.eager
train_loss_results = []
train_accuracy_results = []
num_epochs = 601

# Training loop
for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  
  # Print epoch results
  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

# Convert csv test data to tensorflow dataset
test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size, 
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

# Pack features of each (features,label) pair into the test dataset
test_dataset = test_dataset.map(pack_features_vector)                        

# Test loop
test_accuracy = tfe.metrics.Accuracy()
for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
  
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# Single prediction
predict_dataset = tf.convert_to_tensor([
    [5.5, 2.4, 3.8, 1.1,]
])
predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example prediction: {} ({:4.1f}%)".format(name, 100*p))