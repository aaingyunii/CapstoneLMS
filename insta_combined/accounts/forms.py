from django.contrib.auth.models import User
from django import forms

class SignUpForm(forms.ModelForm):

    password = forms.CharField(label = 'Password', widget = forms.PasswordInput)
    Repeat_password = forms.CharField(label = 'Repeat_password', widget = forms.PasswordInput)
  
    class Meta:
        model = User
        fields = ['username', 'password', 'age', 'gender']

    def clean_Repeat_password(self):
        cd = self.cleaned_data
        if cd['password'] != cd['Repeat_password']:
            raise forms.ValidationError('비밀번호 불일치')
        return cd['Repeat_password']