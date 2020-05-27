from django.urls import path
from .views import BookmarkList, BookmarkCreate, BookmarkDelete, BookmarkDetail, BookmarkUpdate

app_name = 'bookmark'

urlpatterns = [
    path('create/', BookmarkCreate.as_view(), name = 'create'),
    path('delete/<int:pk>', BookmarkDelete.as_view(), name = 'delete'), #pk 특정값 가져오기
    path('update/<int:pk>', BookmarkUpdate.as_view(), name = 'update'),
    path('detail/<int:pk>', BookmarkDetail.as_view(), name = 'detail'),
    path('', BookmarkList.as_view(), name = 'index'),
]