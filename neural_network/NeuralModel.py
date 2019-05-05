import numpy as np
import cv2
from keras.preprocessing import image
import time
#face expression recognizer initialization
from keras.models import model_from_json

class NeuralModel: 
    emotions = ('angry','disgust','fear','happy','sad','surprise','neutral')

    emotion_aggregates_dictionary =	{
        'angry':0,
        'disgust':0,
        'fear':0,
        'happy':0,
        'sad':0,
        'surprise':0,
        'neutral':0
    }

    emotion_by_frame_dict =	{
        'angry':0,
        'disgust':0,
        'fear':0,
        'happy':0,
        'sad':0,
        'surprise':0,
        'neutral':0
    }
 

    def loadModel(self,modelJson,modelWeight):
        self.model = model_from_json(open(modelJson, "r").read())
        self.model.load_weights(modelWeight) #load weights

    def setCascadeClassifier(self,classifierLink):
         self.face_cascade = cv2.CascadeClassifier(classifierLink)
  

    def analyzeVideo(self,videoUrl):

       # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(videoUrl) #process videos
        
        frame = 0
        count = 1
        
        while(True):
          ret, img = cap.read()
          if img is not None:
            img = cv2.resize(img, (640, 360))
                #img = img[0:308,:]

          
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                if w > 130: #trick: ignore small faces
                    cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face
                    
                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
                    
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    
                    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                    
                    #------------------------------
                    
                    predictions = self.model.predict(img_pixels) #store probabilities of 7 expressions
                    max_index = np.argmax(predictions[0])
                    
                    #background of expression list
                    overlay = img.copy()
                    opacity = 0.4
                    cv2.rectangle(img,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
                    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                    
                    #connect face and expressions
                    cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
                    cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)
                    
                    emotion = ""
             
                    
                    for i in range(len(predictions[0])):
                        emotion = "%s %s%s" % (self.emotions[i], round(predictions[0][i]*100, 2), '%')
                    
                        
                        if round(predictions[0][i]*100, 2) > 50:                         
                          
                           self.emotion_by_frame_dict[self.emotions[i]]= self.emotion_by_frame_dict[self.emotions[i]]+1

                      
                        self.emotion_aggregates_dictionary[self.emotions[i]]= self.emotion_aggregates_dictionary[self.emotions[i]]+predictions[0][i]
                        if i != max_index:
                            color = (255,0,0)
                            
                        color = (255,255,255)
                        
                        cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        count+=1
                    #-------------------------
            
            cv2.imshow('img',img)
            frame = frame + 1
            
            
        
          
          
          if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                break
        #kill open cv things
        cap.release()
        cv2.destroyAllWindows()
 

        for i in range (len(self.emotions)):         
          
            self.emotion_aggregates_dictionary[self.emotions[i]]= round(self.emotion_aggregates_dictionary[self.emotions[i]]*100,2) /count          
   
    
    def get_emotion_by_frame(self):
     
            return self.emotion_by_frame_dict
      
    def get_emotion_aggregates(self): 

           return self.emotion_aggregates_dictionary

