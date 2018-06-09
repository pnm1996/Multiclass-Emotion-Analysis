import csv
from keras.models import load_model

model = load_model('my_model.h5')

#clf=joblib.load('rnn.pkl')
X_test=[]
f = open('C:\\Users\\sagar_000\\Desktop\\BE project\\abc.csv', 'r')
reader = csv.reader(f)
for row in reader:
    X_test.append(row[0])


yFit = model.predict(X_test, batch_size=10, verbose=1)
print()
print(yFit)
#emotion=model.predict(X_test)
#print(emotion)