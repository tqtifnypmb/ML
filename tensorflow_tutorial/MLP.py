import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def mlp_model(x, n_input, n_hidden_1, n_hidden_2, n_class):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_class]))
    }

    bias = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_class]))
    }
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), bias['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), bias['b2'])
    layer_out = tf.add(tf.matmul(layer_2, weights['out']), bias['out'])
    
    return layer_out

if __name__ == '__main__':
    n_class = 10

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    X = tf.placeholder('float', shape=[None, 784])
    Y = tf.placeholder('float', shape=[None, 10])
    logits = mlp_model(X, 784, 256, 256, 10)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batch_size = 100
        epoches = 15
        display_step = 1

        for epoch in range(epoches):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))