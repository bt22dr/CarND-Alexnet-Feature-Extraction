import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

nb_classes = 43
num_epochs = 2
batch_size = 32

# TODO: Load traffic signs data.
f = open('train.p', 'rb')
data = pickle.load(f)
print(data.keys())
print(len(data['features']))

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
num_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, shape=(None,32,32,3))
labels = tf.placeholder(tf.int64)
resized = tf.image.resize_images(features, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
print("###", fc7.get_shape())
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss_op = tf.reduce_mean(entropy)
optimizer_op = tf.train.AdamOptimizer().minimize(loss_op)
#train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
pred_op = tf.argmax(probs, axis=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, labels), dtype=tf.float32))

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for e in range(num_epochs):
    # Train
    for i in range(0, num_samples, batch_size):
      start_idx = i
      end_idx = i+batch_size
      _, cost = sess.run([optimizer_op, loss_op], feed_dict={features: X_train[start_idx:end_idx], labels: y_train[start_idx:end_idx]})
      if i % (10*batch_size) == 0:
        print("[epoch:{}, iteration:{}/{}]".format(e, i, num_samples), cost)

    # Evaluation
    total_acc = []
    for i in range(0, num_test_samples, batch_size):
      start_idx = i
      end_idx = i+batch_size
      acc = sess.run(acc_op, feed_dict={features: X_test[start_idx:end_idx], labels: y_test[start_idx:end_idx]})
      total_acc.append(acc)

    print(total_acc)
    print('ACC:', np.mean(total_acc))
