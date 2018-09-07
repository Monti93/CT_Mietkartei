import tensorflow as tf    
import numpy as np
import re
import sys

if len(sys.argv) < 5:
	print("Bitte Miete, Nebenkosten, Quadratmeter und Kueche in dieser Reihenfolge mit der richtigen Formatierung uebergeben")
	exit()

graph = tf.get_default_graph()
sess = tf.Session()
tf.train.import_meta_graph('checkpoints/neural_network_model.ckpt.meta').restore(sess, tf.train.latest_checkpoint('checkpoints/'))
Y_one_hot = graph.get_tensor_by_name("Y_one_hot:0")
Y = graph.get_tensor_by_name("Y:0")
X = graph.get_tensor_by_name("X:0")
prob_keep = graph.get_tensor_by_name("prob_keep:0")

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
    
# helper function that transforms the type of available kitchen based on an ordinal scale (the newer the kitchen the better --> higher value)
def transformKitchen(whichKitchen):
    if whichKitchen == 'Kueche (neu)':
        return 2
    elif whichKitchen == 'Kueche (alt)':
        return 1
    else:
        return 0
    
# helper function to perform min-max scaling of features
def scaleFeatures(features):
    # min-max values taken from scaling computations of training script
    min = np.array([0.,275.5,20.]) 
    max = np.array([2.,1275.5,120.]) 
    for attr in range(min.shape[0]):
        features[attr] = (features[attr] - min[attr]) / (max[attr] - min[attr]) # scale each feature based on min-max normalization
    return features
    
def getTransformedFeaturArray(rentalFee, additionalCosts, squareMeter, whichKitchen):
    return scaleFeatures(np.array([transformKitchen(whichKitchen), transformCosts(rentalFee, additionalCosts), transformSquareMeter(squareMeter)], ))

X_ = getTransformedFeaturArray(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
X_.shape = (1,3)
prob, pred_class = sess.run([Y, Y_one_hot], {X: X_, prob_keep: 1.0})
if pred_class[0] == 1:
    answer = 'ja'
else:
    answer = 'nein'
print("predicted class: '{}' with propability of {}%".format(answer, prob[0, pred_class[0]]*100))

