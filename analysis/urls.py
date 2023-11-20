from django.urls import path
from .import views
urlpatterns=[
    path("Signup",views.SignupPage,name="SignupPage"),
    path("",views.LoginPage,name="LoginPage"),
    path("Logout",views.LogoutUser,name="Logout"),
    path('Home',views.home,name='home'),
    path('Technical-analysis',views.Technical_analysis,name='Technical-analysis'),
    path('Business-analysis',views.Business_analysis,name='Business-analysis'),
    path('Report-analysis',views.Report_analysis,name='Report-analysis'),
    path('Technical-data',views.Technical_data,name='Technical-data'),
    path('filter_data/', views.filter_data, name='filter_data'),
    # path('Business-analysis/<str:bank_identifier>/', views.bank_reviews_view, name='bank_reviews')
]