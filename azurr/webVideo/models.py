from django.db import models
from django.conf import settings

# Create your models here.

class Eve(models.Model):
    name = models.IntegerField()
    result = models.IntegerField()
    state = models.CharField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True,null=True,blank=True)


# class Photh(models.Model):
#     number = models.IntegerField()
#     image = models.ImageField(blank=True)
#     pub_date = models.DateTimeField(auto_now_add=True)