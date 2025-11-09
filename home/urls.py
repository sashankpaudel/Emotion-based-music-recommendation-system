from django.urls import path
from home import views

app_name = "home"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/", views.api, name="api"),
    path("webcam/", views.webcam_data, name="webcam_data"),
    path("getsong/", views.getsong, name="getsong"),
    path("songapi/", views.songapi, name="songapi"),
    path("login/", views.login, name="login"),
    # path("songs/", views.song, name="song")
    # path("get_emotion/", views.get_emotion(), name="get_emotion")
]