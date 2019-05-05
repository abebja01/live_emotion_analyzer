import sys
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt


class ModelBuilder:
            #------------------------------
        x_train, y_train, x_test, y_test = [], [], [], []
        #cpu - gpu configuration
        config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)
        #------------------------------
        #variables
        num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
        batch_size = 256
        epochs = 5
        #------------------------------
        #read kaggle facial expression recognition challenge dataset (fer2013.csv)
        #https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

        def generate_model(self,training_data_location):
            
                with open(training_data_location) as f:
                    content = f.readlines()

                lines = np.array(content)

                num_of_instances = lines.size
                print("number of instances: ",num_of_instances)
                print("instance length: ",len(lines[1].split(",")[1].split(" ")))

                #------------------------------
                #initialize trainset and test set
                

                #------------------------------
                #transfer train and test set data
                for i in range(1,num_of_instances):
                    try:
                        emotion, img, usage = lines[i].split(",")
                        
                        val = img.split(" ")
                            
                        pixels = np.array(val, 'float32')
                        
                        emotion = keras.utils.to_categorical(emotion, self.num_classes)
                    
                        if 'Training' in usage:
                            self.y_train.append(emotion)
                            self.x_train.append(pixels)
                        elif 'PublicTest' in usage:
                            self.y_test.append(emotion)
                            self.x_test.append(pixels)
                    except:
                     print("",end="")

                #------------------------------
                #data transformation for train and test sets
                self.x_train = np.array(self.x_train, 'float32')
                self.y_train = np.array(self.y_train, 'float32')
                self.x_test = np.array(self.x_test, 'float32')
                self.y_test = np.array(self.y_test, 'float32')

                self.x_train /= 255 #normalize inputs between [0, 1]
                self.x_test /= 255

                self.x_train = self.x_train.reshape(self.x_train.shape[0], 48, 48, 1)
                self.x_train = self.x_train.astype('float32')
                self.x_test = self.x_test.reshape(self.x_test.shape[0], 48, 48, 1)
                self.x_test = self.x_test.astype('float32')

                print(self.x_train.shape[0], 'train samples')
                print(self.x_test.shape[0], 'test samples')
                #------------------------------
                #construct CNN structure
                self.model = Sequential()

                #1st convolution layer
                self.model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
                self.model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

                #2nd convolution layer
                self.model.add(Conv2D(64, (3, 3), activation='relu'))
                self.model.add(Conv2D(64, (3, 3), activation='relu'))
                self.model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

                #3rd convolution layer
                self.model.add(Conv2D(128, (3, 3), activation='relu'))
                self.model.add(Conv2D(128, (3, 3), activation='relu'))
                self.model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

                self.model.add(Flatten())

                #fully connected neural networks
                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(0.2))
                self.model.add(Dense(1024, activation='relu'))
                self.model.add(Dropout(0.2))

                self.model.add(Dense(self.num_classes, activation='softmax'))
                #------------------------------
                #batch process
                gen = ImageDataGenerator()
                self.train_generator = gen.flow(self.x_train, self.y_train, batch_size=self.batch_size)

                #------------------------------

                self.model.compile(loss='categorical_crossentropy'
                    , optimizer=keras.optimizers.Adam()
                    , metrics=['accuracy']
                )

        def train_model(self,option):
            
            if option=="all":
                print("\n=====Training model=====")
                print("Beggining training for all dataset... \n")
                self.model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
            elif option=="random":

                print("\n=====Training model=====")
                print("Beggining training for random dataset... \n")
                self.model.fit_generator(self.train_generator, steps_per_epoch=self.batch_size, epochs=self.epochs) #train for randomly selected one
            else:
               
                print("===== WARNING =====\n")
                print("Second parameter must be either: \n\n - 'all' --> train for all dataset \n -'random' --> train for random datasets \n")
                print(" MODEL IS NOT TRAINED , SYSTEM IS TERMINATING \n ")
                print("===== WARNING =====\n")
                sys.exit()
        
        def save_model(self,directory,model_structure_json_name,model_weights_file_name):

                model_json_location = directory+"/"+model_structure_json_name+".json"
                model_weights_file =  directory+"/"+model_weights_file_name+".h5"
                print("===Saving model json and weights===")
                model_json = self.model.to_json()                
                with open(model_json_location, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                self.model.save_weights(model_weights_file)

        
        def evaluate_model(self):

            score = self.model.evaluate(self.x_test, self.y_test)
            print('Test loss:', score[0])
            print('Test accuracy:', 100*score[1])