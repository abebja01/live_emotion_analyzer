from django.shortcuts import render
from django.views.generic import TemplateView
from django.views.generic import ListView
from neural_network.NeuralModel import NeuralModel
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from .models import UploadedFile
from django.http import HttpResponse
import datetime
from django.shortcuts import redirect
from django.core.files.storage import FileSystemStorage
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

# Create your views here.
class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'auth_register.html'

class IndexPageView(TemplateView):        
    template_name = 'index.html'

class UploadedFilesList(ListView):
    model = UploadedFile    

class DashboardPageView(TemplateView):
    template_name = 'dashboard.html'
    
    def get_context_data(self, **kwargs):
       context = super(DashboardPageView, self).get_context_data(**kwargs)
       # here's the difference:
       context['files'] = UploadedFile.objects.filter(user=self.request.user)
       return context

class VideoPageView(TemplateView):
    template_name = 'video.html'

def SubirVideoPageView(request):
    #get uploaded file
    uploaded_file = request.FILES['file']
    #check if there is an user authenticated
    user_is_authenticated = request.user.is_authenticated
    
    if user_is_authenticated:
        auth_user = request.user.username
    #get current date
    current_date = datetime.datetime.now().date()
    #save the uploaded file in media folder
    fs = FileSystemStorage()
    filename = fs.save(uploaded_file.name, uploaded_file)
    #get the url and replace special characters
    uploaded_file_url = "."+fs.url(filename)
    replace_spaces = uploaded_file_url.replace("%20"," ")
    replace_corchete = replace_spaces.replace("%5B","[")
    replace_otro_corchete = replace_corchete.replace("%5D","]")
    #create the instance of UploadedFile
    new_file = UploadedFile(user=auth_user, date = current_date, url=replace_otro_corchete)
    new_file.save()
    #redirect to dashboard
    return redirect('dashboard')

def analyze_method(request):   


    video_path = UploadedFile.objects.get(pk=request.GET['videoid']).url  
    neuralModel = NeuralModel()
    modelJsonLink = './neural_network/data/facial_expression_model_structure.json'
    modelWeightsLink = './neural_network/data/facial_expression_model_weights.h5'
    faceCascadeLink = './neural_network/data/haarcascade_frontalface_default.xml'
    neuralModel.setCascadeClassifier(faceCascadeLink)
    neuralModel.loadModel(modelJsonLink,modelWeightsLink)
    neuralModel.analyzeVideo(video_path)

    emotion_by_frame = neuralModel.get_emotion_by_frame()
    emotions_aggregated = neuralModel.get_emotion_aggregates()
  
  
    
    aggregated_list_keys = [ k for k in emotions_aggregated ]
    agreggated_list_values = [ v for v in emotions_aggregated.values() ]
    y_pos = np.arange(len(aggregated_list_keys))

    by_frame_list_keys = [ k for k in emotion_by_frame ]
    by_frame_list_values = [ v for v in emotion_by_frame.values() ]
    y_pos2 = np.arange(len(by_frame_list_keys))

    plt.figure(1)
    plt.bar(y_pos,agreggated_list_values, align='center', alpha=0.5)
    plt.xticks(y_pos, aggregated_list_keys)
    plt.ylabel('aggregation (scale to 10)')
    plt.title('Emotions Aggregated')
   
    plt.figure(2)
    plt.bar(y_pos2,by_frame_list_values, align='center', alpha=0.5)
    plt.xticks(y_pos2, by_frame_list_keys)
    plt.ylabel('Percentage')
    plt.title('Emotions By Frame')
    
    plt.show()
    K.clear_session()




    return redirect('dashboard')