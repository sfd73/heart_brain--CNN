
#%%
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(units=256, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./Training',
 target_size=(256, 256),
  batch_size=32,
   class_mode='binary')


test_set = test_datagen.flow_from_directory('./Testing',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary')

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
path = (r"C:\Users\seraj\Desktop\ALL-projrct\simple CNN\Training\brain\brain.jpg ")

img = mpimg.imread(path)
plt.imshow(img)
plt.axis('off')


#%%
class_dict = training_set.class_indices
print(class_dict)

#%%
model.fit(training_set,
    epochs=15,
    validation_data=test_set)


#%%


#%%

#plt.axis('off')


import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

img = cv2.imread('2222.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (128,128))
plt.imshow(resize.numpy().astype(int))
plt.show()



result = model.predict(np.expand_dims(resize/255, 0))
if result > 0.5:
    print('heart ')

else:
    print(' brain ')  


#model.save('brain_heart_classifier_v1.h5')






# %%
