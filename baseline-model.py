import sckit
import os


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from pre_process import AudioPreProcessor 


# Loading in the processor (this may be changed in the future to allow for path changes when loading)
ap = AudioPreProcessor()
ap.load(path='training')

train_mfcc = ap.show_mfccs()
train_labels = ap.show_labels()

# Resetting mfcc lists to be used for validating the model
ap.clear()
ap.load(path='validation')

validate_mfcc = ap.show_mfccs()
validate_labels = ap.show_labels()

# Resetting mfcc list to be used for testing the model
ap.clear()
ap.load(path='testing')


test_mfcc = ap.show_mfccs()
test_labels = ap.show_labels()


scaler = StandardScaler()
train_mfcc_scaled = scaler.fit_transform(train_mfcc)
test_mfcc_scaled = scaler.transform(test_mfcc)


# So in this case the data is already seperate so i do not need to process it to that extent
baseline = LogisticRegression(max_iter=1000).fit(train_mfcc_scaled, train_labels)

# However, this need to be a conglamorate of both sets so that it correctly associates the assumptions.
predictions = baseline.predict(test_mfcc_scaled)

accuracy = accuracy_store(test_labels, predictions)

precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, predictions, average='weighted')

# now for model 'correctness', the following features are used:
# Precision
# Accuracy
# Recall
# F-scores

print(f"The accuracy of this model is: {accuracy:.2f}")
print(f"The precision of this model is: {precision:.2f}")
print(f"The recall value for this model is: {recall:.2f}")
print(f"The fscore for this model is: {fscore:.2f}")


