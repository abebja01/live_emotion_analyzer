# emotionanalyzer_app/urls.py
from django.urls import path

from .views import IndexPageView
from .views import DashboardPageView
from .views import VideoPageView
from .views import SubirVideoPageView,analyze_method
from . import views

urlpatterns = [
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('', IndexPageView.as_view(), name='index'),
    path('analyze', analyze_method, name='test'),
    path('dashboard', DashboardPageView.as_view(), name='dashboard'),
    path('video', VideoPageView.as_view(), name='video'),
    path('subir_video', SubirVideoPageView, name='subir_video')
]