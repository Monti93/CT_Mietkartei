
# coding: utf-8

# In[275]:


import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import csv
import re
import math


# In[276]:


number_of_features = 2  # used for shaping training and test input data
number_of_labels = 2    # used for shaping the one-hot encoded label for predicted classes
batch_size = 100        # used for batch learning
generations=5000        # number of generations used for learning


# In[277]:


# helper function that transforms all monthly costs into a number
def transformCosts(rentalFee, additionalCosts):
    fee      = re.search('([0-9]*)-([0-9]*)', rentalFee) # extract the two numbers in range of rentalFee i.e. "251-300" ==> 251 and 300
    addCosts = re.search('(unter |ueber |[0-9]*)-?([0-9]*)', additionalCosts) # extract the two numbers or the single number of additional costs i.e. "50-100" ==> 50 and 100 or "unter 50" ==> 50
    
    # compute the middle between those two numbers or simply take the only given number if only one exists
    midFee = (int(fee.group(1)) + int(fee.group(2))) / 2
    try:
        # works if both groups are numbers
        midAddCosts = (int(addCosts.group(1)) + int(addCosts.group(2))) / 2
    except ValueError:
        # happens if first group is no number
        midAddCosts = int(addCosts.group(2))
    return midFee + midAddCosts

# helper function that transforms the range of square meters to the mid number
def transformSquareMeter(squareMeter):
    # works similar to additional cost extraction in transformCosts()
    sm = re.search('(bis |ueber |[0-9]*)-?([0-9]*)', squareMeter)
    try:
        return (int(sm.group(1)) + int(sm.group(2))) / 2
    except ValueError:
        return int(sm.group(2))
    
# helper function that transforms the classes "ja" and "nein" to one-hot encoded array 
def transformPrediction(target):
    if(target == "ja"):
        return np.array([0,1])
    else:
        return np.array([1,0])        


# In[278]:


# helper function to perform min-max scaling of features
def scaleFeatures(features):
    min = features.min(0) # get min values for every feature column-wise
    max = features.max(0) # get max values for every feature column-wise
    for row in range(features.shape[0]):
        for attr in range(min.shape[0]):
            features[row, attr] = (features[row, attr] - min[attr]) / (max[attr] - min[attr]) # scale each feature based on min-max normalization
    return features

def transformCsvToFeatures(csvUrl):
    with open(csvUrl, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None) # skip the header
        X = np.array([]) # matrix for features
        Y = np.array([]) # one-hot encoded matrix for labels (classes)
        rowCount = 0
        for row in reader:
            #print(row[8], row[9], row[21], row[22])
            X = np.append(X, np.array([transformCosts(row[8], row[9]), transformSquareMeter(row[21])]))
            Y = np.append(Y, transformPrediction(row[22]))
            rowCount += 1
        X.shape = (rowCount, number_of_features)
        Y.shape = (rowCount, number_of_labels)
        return (scaleFeatures(X), Y) 
        #print(X)
        #print(Y)


# In[279]:


train_Features = transformCsvToFeatures('Wohnungskartei_Muster_Master_4_S.csv')
test_Features = transformCsvToFeatures('Wohnungskartei_Muster_Master_6_S_teach.csv')
# features can be accessed via index 0, labels via index 1
# i.e. train_Features[0] --> train features
# i.e. train_Features[1] --> train labels


# In[280]:


neurons_first_layer  = 100 # number of neurons used in first hidden layer 
neurons_second_layer = 50  # number of neurons used in second hidden layer
neurons_third_layer  = 25  # number of neurons used in third hidden layer


# In[281]:


X = tf.placeholder(tf.float32, [None, number_of_features])  # Tensoflow placeholder for training samples
Y_ = tf.placeholder(tf.float32, [None, number_of_labels])   # Tensoflow placeholder for training labels (the provided ones as teacher)
learning_rate = tf.placeholder(tf.float32)                  # Tensoflow placeholder for learning rate controls how big adjustment steps for backpropagation should be
prob_keep = tf.placeholder(tf.float32)                      # Tensoflow placeholder for the propability for neurons to be kept in each layer (dropout) to prevent overfitting


# In[282]:


# weights will be initialized with small random values between -0.2 and +0.2
# RELU will be used as activation function, it is common then to initialize biases with small positive values for example 0.1 = tf.ones([...])/10

# weights and biases from input to first hidden layer
W1 = tf.Variable(tf.truncated_normal([number_of_features, neurons_first_layer], stddev=0.1), name="W1")  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([neurons_first_layer])/10, name="B1")

# weights and biases from first hidden to second hidden layer
W2 = tf.Variable(tf.truncated_normal([neurons_first_layer, neurons_second_layer], stddev=0.1), name="W2")
B2 = tf.Variable(tf.ones([neurons_second_layer])/10, name="B2")

# weights and biases from second hidden to third hidden layer
W3 = tf.Variable(tf.truncated_normal([neurons_second_layer, neurons_third_layer], stddev=0.1), name="W3")
B3 = tf.Variable(tf.ones([neurons_third_layer])/10, name="B3")

# weights and biases from third hidden to output layer
W4 = tf.Variable(tf.truncated_normal([neurons_third_layer, number_of_labels], stddev=0.1), name="W4")
B4 = tf.Variable(tf.ones([number_of_labels])/10, name="B4")


# In[283]:


# The model output with dropout at each layer
# RELU as activation function

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, prob_keep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, prob_keep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, prob_keep)

# last layer with softmax to get continuous values between 0 an 1
Ylogits = tf.matmul(Y3d, W4) + B4
Y = tf.nn.softmax(Ylogits)

# argmax returns the highest value representing the most propable class regarding the one-hot encoded labeling (used for model output later on)
Y_one_hot = tf.argmax(Y, 1)


# In[284]:


# cross-entropy will be used as loss function ==> -sum(Y_i * log(Yi)) 
# the value will be normalised for batches of batch_size entries as initialized in the beginning
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of current model between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[285]:


# this function can be called in a loop to train the model and store evaluation results
def training_step(i, update_test_data, update_train_data, last_step):
    rand_index = np.random.choice(len(train_Features[0]), size=batch_size) # returns random indizes with shape of batch size
    batch_X = train_Features[0][rand_index] # select features based on random indices
    batch_Y = train_Features[1][rand_index] # select labels based of random indices
    
    # adjust the learning rate based on learning progress --> it becomes smaller each generation
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    decayed_learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    
    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, prob_keep: 0.75, learning_rate: decayed_learning_rate})

    a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, prob_keep: 1.0})

    if update_train_data:
        # print training evaluation values
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (learning rate:" + str(decayed_learning_rate) + ")")

    if update_test_data:
        # print test evaluation values
        train_loss.append(c)
        train_acc.append(a)
        a, c = sess.run([accuracy, cross_entropy], {X: test_Features[0], Y_: test_Features[1], prob_keep: 1.0})
        test_acc.append(a)
        print(str(i) + ": ********* epoch " + str(i*100//test_Features[0].shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))


# In[288]:


# Adam Optimizer is a backpropagation algorithm that converges faster than standard backpropagation
# it will try to minimize cross entropy as loss function with gradient descent given a certain learning rate
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# initilize the TensorFlow graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# these arrays will be filled with evaluation values in function training_step()
train_loss = []
train_acc = []
test_acc = []

# start training
# print training evaluation every 20 steps
# print test evaluation every 100 steps
for i in range(generations+1): training_step(i, i % 100 == 0, i % 20 == 0, i == generations)


# In[289]:


# first plot: Plot the loss and accuracies of training and test data
eval_indices = range(0, generations+eval_every, 100)
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# second plot: Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[290]:


# the following builder can be used for creating servable models that can be used for production
builder = tf.saved_model.builder.SavedModelBuilder("neural_network_model_only_cost_and_meter")
builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING], {
            "mietkartei": tf.saved_model.signature_def_utils.predict_signature_def(
                inputs= {"features": X, "prob_keep": prob_keep},
                outputs= {"label_index": Y_one_hot, "propability": Y})
            })
builder.save()


# In[291]:


# the following saver stores the all variable values (weights and biases) of the trained model for restoring the model anytime
saver = tf.train.Saver()
save_path = saver.save(sess, "checkpoints/only_cost_and_meter/neural_network_model.ckpt")
print("Model saved in path: %s" % save_path)

