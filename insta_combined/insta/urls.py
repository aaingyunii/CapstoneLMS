from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('accounts/', include('accounts.urls')),
    path('bookmark/', include('bookmark.urls')),
    path('admin/', admin.site.urls),
    path('', include('photo.urls')), #빈칸이 이쓰면 include로 보내라.
]
