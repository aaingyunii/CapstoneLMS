from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.


# class User_info(AbstractUser):

#     gender = models.CharField(max_length=30)  # True for male and False for female
#     age=models.IntegerField()

# class Account(models.Model):
#     username = models.CharField(max_length = 20)
#     password = models.CharField(max_length = 50)
#     # age = models.IntegerField()
#     # gender = models.CharField(max_length=10)
#     created_at = models.DateTimeField(auto_now_add = True,null=True,blank=True)
