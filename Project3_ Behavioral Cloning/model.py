import csv
import cv2
import numpy as np

def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are 
    received in RGB)
    '''
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return new_img

def shuffle_dataset(x,y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    return x, y


lines = []
with open('../Collected Data/driving_log.csv') as csvfile:#('../data/driving_log.csv') as csvfile:#data-->Collected Data
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]#source_path.split('/')[-1]
    current_path = '../Collected Data/IMG/' + filename#'../data/IMG/' + filename
    image = cv2.imread(current_path)
    new_img = preprocess_image(image)#cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    images.append(new_img)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []    
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

#Shuffle the inputs
augmented_images, augmented_measurements = shuffle_dataset(np.asarray(augmented_images), np.asarray(augmented_measurements))

split_rate = 0.8
nsample = np.shape(augmented_images)[0]

X_train = augmented_images[ : int(nsample*split_rate)]
y_train = augmented_measurements[ : int(nsample*split_rate)]

X_test = augmented_images[int(nsample*split_rate) : ]
y_test = augmented_measurements[int(nsample*split_rate) : ]


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D


model = Sequential()
#Normalize inputs
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))

#Crop images
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Build CNN network
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.7))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=32, validation_split=0.2, 
          shuffle=True, nb_epoch=20)

test_loss = model.evaluate(X_test, y_test)
print('test loss is : ', test_loss)

model.save('model.h5')

#exit()
