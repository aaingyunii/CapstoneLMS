from django.urls import path
from .views import PhotoCreate, PhotoDelete, PhotoList, PhotoDetail, PhotoUpdate 
from photo.views import webcam_test, mic_test, home, index, room


app_name= 'photo'
urlpatterns = [
    path('create/', PhotoCreate.as_view(), name='create'),# as view를 붙여야 generic이용가능
    path('delete/<int:pk>/', PhotoDelete.as_view(), name='delete'),
    path('update/<int:pk>/', PhotoUpdate.as_view(), name='update'),
    # path('/main/',main,name='detail'),
    path('detail/<int:pk>/', PhotoDetail.as_view(), name='detail'),
    path('', PhotoList.as_view(), name='index'),
    path('webcam_test/', webcam_test),
    path('mic_test/',mic_test),
    path('home/', home),
    path('index/',index),
    path('index/<str:room_name>/', room, name='room')
]

from django.conf.urls.static import static
from django.conf import settings

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)