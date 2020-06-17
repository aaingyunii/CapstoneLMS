from django.urls import path
from . import consumers

websocket_urlpatterns = [

    path('ws/index/<str:room_name>/', consumers.ChatConsumer),

]


