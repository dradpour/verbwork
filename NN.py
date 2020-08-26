#Authors: Vinay Ashokkumar

import tensorflow as tf
import numpy as np
import pickle

train_count=200
test_count=100


#Function to define the labels in the form of one-hot vectors.
def getonehot(label):
    if(label==0):
        return [1,0]
    elif(label==1):
        return [0,1]
    else:
        print("Invalid Label")
        
#Function to get pickled data sets of features calculated from the preprocessed data dictionaries. 
def getdata(rating):
    data=pickle.load(open("C:\\Users\\Vinay\\Desktop\\CSE 674\\yelp data\\" + str(rating) + "star_frames.p", "rb"))
    result = []
    labels = []
    for entry in data:
        labels.append(getonehot(entry["label"]))#Create the Label lists of the data set. '0' = Non-sarcastic, and '1' = Sarcastic.
        del entry["label"]
        result.append([int(value) for key,value in entry.items()])#Convert all the dictionaries in to lists 
    result = np.array(result, dtype="float32")
    labels = np.array(labels, dtype="float32")
    return (result[0:train_count], labels[0:train_count]), (result[train_count:train_count + test_count], labels[train_count:train_count + test_count])

train, test = getdata(5)#Division of testing and training data sets for each star reviewed pickle file. 
#print len(train[1][0])
#print test



# Network Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 50
display_step = 1
iterv  = -batch_size

# Network Parameters
n_hidden_1 = 15 # 1st layer number of features
n_hidden_2 = 15 # 2nd layer number of features
n_input = 15 #15 input nodes representing 15 features
n_classes = 2 #2 output classes

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def nextbatch():
    global iterv
    iterv = iterv + batch_size
    return train[0][iterv:iterv+batch_size], train[1][iterv:iterv+batch_size]
    
    
    

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_count/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = nextbatch()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test[0], y: test[1]}))



#References for Code Construction:
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb