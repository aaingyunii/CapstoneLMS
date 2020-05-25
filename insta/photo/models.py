from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse

class Photo(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user')
    text = models.TextField(blank=True)
    # video = models.ImageField(blank=True, upload_to= 'timeline_photo/%Y/%m/%d')#d알아서 날짜 추가.
    image = models.ImageField(upload_to= 'timeline_photo/%Y/%m/%d')#d알아서 날짜 추가.
    created = models.DateTimeField(auto_now_add=True)#지금시간 생성
    updated = models.DateTimeField(auto_now=True)#한번 업데이트
    
    def __str__(self):
        return "text : " +self.text
    
    class Meta:
        ordering = ['-created']

    def get_absolute_url(self):
        return reverse('photo:detail', args=[self.id])