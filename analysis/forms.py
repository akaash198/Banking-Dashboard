from django import forms
from .models import Bank
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class BankSelectionForm(forms.Form):
    bank = forms.ModelChoiceField(queryset=Bank.objects.all())


class CreateUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class ProfileForm(forms.Form):
    bio = forms.CharField(max_length=500, required=False)
    avatar = forms.ImageField(required=False)
