from django.shortcuts import render, redirect
from django.contrib.auth.models import User
# from .forms import SignUpForm
from django.contrib import auth
# from .models import Account
from datetime import datetime

def signup(request):
    if request.method == 'POST':
        if request.POST['password1'] == request.POST['password2']:
            user = User.objects.create_user(
                username=request.POST['username'], password=request.POST['password1'],
                    first_name=request.POST['first_name'],last_name=request.POST['last_name'],email=request.POST['email'])
            
            auth.login(request, user)
            return render(request, 'accounts/signup_complete.html')
        return render(request, 'accounts/signup.html')

    return render(request, 'accounts/signup.html')

def login(request):
    if request.method =='POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username = username, password = password)
        if user is not None:
            auth.login(request, user)
            return render(request, 'photo/home.html')
        else:
            return render(request, 'accounts/login.html', {'error': 'us or apssis not coree'})
    else:
        return render(request, 'accounts/login.html')
def logout(request):
    auth.logout(request)
    return render(request, 'photo/home.html')


# def signup(request):
#     if request.method == 'POST':
#         signup_form = SignUpForm(request.POST)
        
#         if signup_form.is_valid():
#             user_instance = signup_form.save(commit=False)
#             user_instance.set_password(signup_form.cleaned_data['password'])
#             user_instance.save()
#             return render(request, 'accounts/signup_complete.html', {'username':user_instance.username})
#     else:
#         signup_form = SignUpForm()

#     return render(request, 'accounts/signup.html', {'form':signup_form.as_p})