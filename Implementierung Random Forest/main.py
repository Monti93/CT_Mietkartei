import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Files Paths
TEST_DATA_CSV = './Wohnungskartei/Wohnungskartei_Muster_Master_4_S.csv'
TRAIN_DATA_CSV = './Wohnungskartei/Wohnungskartei_Muster_Master_6_S_teach.csv'

# Constants 
DEFAULT_SEPEARTOR = ";"
ATTRIBUTES = ["Zimmerzahl", "Stockwerk", "Heizung", "Hausmeister", "Kindergarten", "Schule", "S-Bahn", "Garage", "Miete", "Nebenkosten", "Alter", "Aufzug", "Lage", "Entfernung", "Kaution", "Kueche", "Bad", "Balkon", "Terrasse", "Kehrwoche", "Moebliert", "Quadratmeter"]
LABEL_COLUMN = 'S'
POSITIVE_LABEL = 'ja'

STANDARD_SCALER = StandardScaler()

def prepareDataSet(file):
    # Read CSV file into panda.dataFrame
    df = pd.read_csv(file, sep = DEFAULT_SEPEARTOR)
    # One-Hot-Encoding Attributes
    df = pd.get_dummies(df, columns = ATTRIBUTES)
    # Replace 'ja' und 'nein' by 0 and 1
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: 1 if x == POSITIVE_LABEL else 0)
    x = df.iloc[:, 1:121].values
    y = df.iloc[:, 1].values   
    return {'x': x, 'y': y}

trainingData = prepareDataSet(TRAIN_DATA_CSV)
testData = prepareDataSet(TEST_DATA_CSV)

# scale values
x_train = STANDARD_SCALER.fit_transform(trainingData['x'])
x_test = STANDARD_SCALER.transform(testData['x'])

# generate RandomForestClassifier
best_accuracy = 0
for x in range (1, 10000):
    classifier = RandomForestClassifier(n_estimators=x, random_state=0)
    classifier.fit(x_train, trainingData['y'])
    y_pred = classifier.predict(x_test)

    current_accuracy = accuracy_score(testData['y'], y_pred)
    if (current_accuracy > best_accuracy):
        print "Found best accuracy: " + str(current_accuracy) + "! With number of trees: " + str(x) + "."
        best_accuracy = current_accuracy

# For furher result information
#print(confusion_matrix(testData['y'],y_pred))  
#print(classification_report(testData['y'],y_pred))  
#print(accuracy_score(testData['y'], y_pred))  