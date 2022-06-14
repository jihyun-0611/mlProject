import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report



input = pd.read_csv("feature_target.csv")

feature_input = input[['dedicated_area', 'deposit', 'rent', 'new_construction', 'cvs_distance', 'station_distance', 'hospital_distance']].to_numpy()
feature_target = input['suitability'].to_numpy()


train_input, test_input, train_target, test_target = train_test_split(feature_input, feature_target)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

mlp = Sequential()

mlp.add(Dense(7, activation='relu', input_shape=(7,),
              kernel_initializer='glorot_uniform', bias_initializer='zeros'))
mlp.add(Dense(10, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros'))
mlp.add(Dense(5, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros'))
mlp.add(Dense(5, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros'))
mlp.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros'))

mlp.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
hist = mlp.fit(train_scaled, train_target, batch_size=512, epochs=40, validation_data= (test_scaled,test_target))


res = mlp.evaluate(test_scaled, test_target, verbose=0)
print(res)



import matplotlib.pyplot as plt

#정확률 곡선
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid()
plt.show()

#손실함수 곡선
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show()

#모델 평가
pred = mlp.predict(test_scaled)
preds_1d = pred.flatten()
pred_class = np.where(preds_1d > 0.5, 1, 0)
target_names = ['부적합', '적합']
print(classification_report(test_target, pred_class, target_names = target_names))