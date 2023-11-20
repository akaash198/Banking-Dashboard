from django.shortcuts import render
from google_play_scraper import app, Sort, reviews
from .models import Bank, BankReviewData, TechnicalAnalysisData
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from django.http import JsonResponse
from textblob import TextBlob
from datetime import datetime, timedelta
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from .forms import CreateUserForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from collections import defaultdict
from django.http import HttpResponse


# Create your views here.


def SignupPage(request):
    form = CreateUserForm()
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account was created for ' + user)
            return redirect('LoginPage')
    context = {'form': form}     
    return render(request, 'Signup.html', {'form': form})


def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.info(request, 'Username OR password is incorrect')
    context = {}
    return render(request, 'Login.html', context)


def LogoutUser(request):
    logout(request)
    return redirect('LoginPage')

@login_required(login_url='LoginPage')
def home(request):
    return render(request,'Home.html')

def fetch_selected_bank():
    try:
        selected_bank = Bank.objects.filter(is_active=True).first()
        return selected_bank
    except Bank.DoesNotExist:
        return None
    
def fetch_bank_reviews_data():
    try:
        selected_bank = Bank.objects.filter(is_active=True).first()
        bank_review_data = BankReviewData.objects.filter(bank=selected_bank, is_active=True).first()
        technical_analysis_data = TechnicalAnalysisData.objects.filter(bank=selected_bank, is_active=True).first()

        return bank_review_data , technical_analysis_data
    except BankReviewData.DoesNotExist:
        return None



def fetch_bank_reviews(selected_bank):
    if selected_bank:
        try:
            user_reviews, _ = reviews(
                app_id=selected_bank.app_id,
                lang="en",
                sort=Sort.NEWEST,
                count=5   
            )
            return user_reviews
        except Exception as e:
            print(f"Error fetching reviews: {e}")
    return [] 

def format_amount(amount):
    # Check if the amount is greater than or equal to 1 million
    if abs(amount) >= 1_000_00:
        # Format the amount in lakhs 
        formatted_amount = f"{amount / 1_000_00:.2f} L"
    else:
        # If the amount is less than 1 million, simply format it with two decimal places
        formatted_amount = f"{amount:.2f}"

    return formatted_amount

def fetch_app_details(selected_bank):
    if selected_bank:
        app_id = selected_bank.app_id  # Assuming selected_bank.app_id contains the package name
        # Fetch app details
        app_info = app(app_id)

        # Extract the desired fields
        app_name = app_info['title']
        overall_rating = round(app_info['score'], 2)  # Rounded to 2 decimal places
        number_of_reviews = app_info['ratings']   # Rounded to 2 decimal places
        downloads = app_info['installs']
        content_rating = app_info['contentRating']

        # Calculate the "Rated for" information based on content rating
        if content_rating == 'Everyone':
            rated_for = '3+'
        elif content_rating == 'Everyone 10+':
            rated_for = '10+'
        elif content_rating == 'Teen':
            rated_for = '13+'
        elif content_rating == 'Mature 17+':
            rated_for = '17+'
        else:
            rated_for = 'Not specified'

        app_name = {
            'name': app_name,
            
        }

        app_details_data = [ 
        {
            'name': overall_rating,
            'text': "Overall Rating",
            'icon': 'ratings.svg'
        },
        {
            'name': format_amount(number_of_reviews),
            'text': "Number of Reviews",
            'icon': 'feedback.svg'
        },
        {
            'name': downloads,
            'text': "Downloads",
            'icon': 'download.svg'
        },
         
        {
            'name': rated_for,
            'text': "Rated For",
            'icon': 'rated.svg'
        }
        ]

        return app_details_data
    else:
        return None

def read_bank_review_data(bank_review_data):
    if bank_review_data:
        try:
            df = pd.read_csv(bank_review_data.csv_file)
            return df
        except Exception as e:
            print(f"Error reading csv file: {e}")
    return None


# def read_technical_analysis_data(technical_analysis_data):
#     if technical_analysis_data:
#         try:
#             df = pd.read_csv(technical_analysis_data.analysis_file)
#             return df
#         except Exception as e:
#             print(f"Error reading csv file: {e}")
#     return None




def read_technical_analysis_data(technical_analysis_data, start_date=None, end_date=None):
    if technical_analysis_data:
        try:
            df = pd.read_csv(technical_analysis_data.analysis_file)

            if start_date and end_date:
                # Convert 'Date and Time' column to datetime
                df['Date and Time'] = pd.to_datetime(df['Date and Time'])

                # Filter the DataFrame based on the specified date range
                filtered_df = df[(df['Date and Time'] >= start_date) & (df['Date and Time'] <= end_date)]
                return filtered_df

            return df
        except Exception as e:
            print(f"Error reading csv file: {e}")

    return None



    





       
    






    






def setiment_context(request):
    nltk.download('vader_lexicon')
    df = read_bank_review_data(fetch_bank_reviews_data()[0])
    print(df)
    analyzer = SentimentIntensityAnalyzer()
    # Function to get sentiment polarity
    def get_sentiment_polarity(text):
        if isinstance(text, str):  # Check if it's a string
            sentiment_scores = analyzer.polarity_scores(text)
            if sentiment_scores['compound'] >= 0.05:
                return 'Positive'
            elif sentiment_scores['compound'] <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
    # Apply sentiment analysis to the "content" column
    df['sentiment'] = df['content'].apply(get_sentiment_polarity)
    # Get sentiment counts
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    return sentiment_counts


def perform_lda_topic_modeling(reviews, num_topics=5, num_words=10):
    """
    Perform Latent Dirichlet Allocation (LDA) topic modeling on a collection of text reviews.

    Parameters:
        reviews (list of str): List of text reviews for topic modeling.
        num_topics (int): Number of topics to extract (default is 5).
        num_words (int): Number of words to display per topic (default is 10).

    Returns:
        data (dict): A dictionary containing topics and coherence score.
    """

    # Tokenize the reviews
    tokenized_reviews = [review['content'].split() for review in reviews]

    # Create a dictionary and a document-term matrix
    dictionary = corpora.Dictionary(tokenized_reviews)
    corpus = [dictionary.doc2bow(review) for review in tokenized_reviews]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Get the topics and their top words
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)

    # Compute the coherence score to evaluate the model
    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    data = {
        'topics': topics,
        'coherence_score': coherence_score
    }

    return data
 

def top_negative_reviews(request):
 
    df_reviews = read_bank_review_data(fetch_bank_reviews_data()[0])
    df_reviews['content'].fillna('', inplace=True)

    # Define the possible related words for each issue
    possible_related_words = {
        'Registration': ['register', 'sign up', 'signup', 'enroll', 'create account'],
        'Login': ['log in', 'signin', 'sign in', 'authentication', 'access'],
        'Transactions': ['transaction', 'payment', 'purchase', 'buy', 'order'],
        'Transfer': ['transfer', 'send money', 'move funds', 'transaction', 'transfer funds'],
        'Balance': ['balance', 'balances', 'money', 'funds', 'cash'],
        'Deposit': ['deposit', 'add money', 'put money', 'credit'],
        'Withdrawal': ['withdraw', 'withdrawal', 'remove money', 'debit'],
        'Account': ['account', 'accounts', 'profile', 'bank account', 'card'],
        'Card': ['card', 'credit card', 'debit card', 'cards'],
        'Rewards': ['reward', 'rewards', 'points', 'cashback', 'benefits'],
    }

    # Initialize a dictionary to store the top 5 negative reviews for each issue along with their thumbs-up counts
    negative_reviews_by_issue = {issue: [] for issue in possible_related_words.keys()}
    thumbs_up_count_by_issue = {issue: [] for issue in possible_related_words.keys()}  # Separate dictionary for thumbs-up counts

    # Find and store the top 5 negative reviews with the highest thumbs-up counts for each issue and related words
    for issue, related_words in possible_related_words.items():
        for keyword in [issue] + related_words:
            # Filter reviews containing the keyword
            filtered_reviews = df_reviews[df_reviews['content'].str.lower().str.contains(keyword.lower())]
            # Sort by thumbs-up count (highest count first)
            sorted_reviews = filtered_reviews.sort_values(by='thumbsUpCount', ascending=False).head(5)
            # Append the top 5 reviews with the highest thumbs-up counts to the list for the current issue
            negative_reviews_by_issue[issue].extend(sorted_reviews['content'].tolist())
            # Append the thumbs-up counts for the top reviews
            thumbs_up_count_by_issue[issue].extend(sorted_reviews['thumbsUpCount'].tolist())

    # Ensure that only the top 5 negative reviews are kept for each issue
    for issue in negative_reviews_by_issue.keys():
        negative_reviews_by_issue[issue] = negative_reviews_by_issue[issue][:5]
        thumbs_up_count_by_issue[issue] = thumbs_up_count_by_issue[issue][:5]

    # Create the dataset
    data = {
        'issues': list(negative_reviews_by_issue.keys()),
        'reviews': negative_reviews_by_issue,
        'thumbs_up_counts': thumbs_up_count_by_issue
    }

    for issue in data['issues']:
        data['reviews'][issue] = sorted(
            data['reviews'][issue],
            key=lambda review: data['thumbs_up_counts'][issue][data['reviews'][issue].index(review)],
            reverse=True
        )

    return data







   
def sentiment_overtime(request):
    nltk.download('vader_lexicon')
    df = read_bank_review_data(fetch_bank_reviews_data()[0])
    analyzer = SentimentIntensityAnalyzer()
    df['at'] = pd.to_datetime(df['at'])
    df['month'] = df['at'].dt.strftime('%b')
    
    # Define a function to get sentiment polarity while handling non-textual data
    def get_sentiment_polarity(text):
        if isinstance(text, str):
            return analyzer.polarity_scores(text)['compound']
        else:
            return 0.0  # Return a neutral sentiment score for non-textual data
    
    # Apply sentiment analysis to the "content" column, handling non-textual data
    df['sentiment_score'] = df['content'].apply(get_sentiment_polarity)
    
    # Calculate the mean sentiment score for each month
    monthly_sentiment = df.groupby('month')['sentiment_score'].mean().reset_index()
    
    # Create data for the chart
    labels = []  # Time labels (e.g., review dates)
    sentiment_scores = []  # Sentiment scores over time

    for index, row in monthly_sentiment.iterrows():
        labels.append(row['month'])
        sentiment_scores.append(row['sentiment_score'])

    # Create the dataset
    data = {
        'labels': labels,
        'sentiment_scores': sentiment_scores
    }
    return data





def app_version_overtime(request):
    df = read_bank_review_data(fetch_bank_reviews_data()[0])
    # Convert the 'at' column to datetime64
    df['at'] = pd.to_datetime(df['at'])
    # Extract the month and year from the 'at' column
    df['month'] = df['at'].dt.strftime('%b')

    # Group by month and app version, and calculate the average sentiment score for each combination
    def calculate_sentiment_scores(text_list):
        text = ' '.join([str(item) for item in text_list])  # Convert items to strings
        return SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    
    monthly_app_version_sentiment = df.groupby(['month', 'appVersion'])['content'].apply(calculate_sentiment_scores).reset_index()
    
    # Rename the columns
    monthly_app_version_sentiment.columns = ['month', 'app_version', 'average_sentiment_score']

    # Group by month and calculate the average sentiment score for each month
    monthly_sentiment = monthly_app_version_sentiment.groupby(['month', 'app_version'])['average_sentiment_score'].mean().reset_index()

    # Create data for the chart
    labels = []  # Labels with both month and app version
    sentiment_scores = []  # Average sentiment scores for each month

    for index, row in monthly_sentiment.iterrows():
        label = f"{row['month']} - {row['app_version']}"  # Combine month and app version
        labels.append(label)
        sentiment_scores.append(row['average_sentiment_score'])

    # Create the dataset
    data = {
        'labels': labels,
        'sentiment_scores': sentiment_scores
    }
    
    return data



@login_required(login_url='LoginPage')
def Business_analysis(request):
    selected_bank = fetch_selected_bank()
    bank_reviews = fetch_bank_reviews(selected_bank)
    bank_logo = selected_bank.logo.url
    app_detail = fetch_app_details(selected_bank)
    topic_modeling = perform_lda_topic_modeling(bank_reviews)
    top_negative_reviews_data = top_negative_reviews(selected_bank)
 
    df = pd.DataFrame(np.array(bank_reviews),columns=['review'])
    df = df.join(pd.DataFrame(df.pop('review').tolist()))
    df = pd.DataFrame(bank_reviews[:5])
    context = {
        'bank': selected_bank,
        'reviews': bank_reviews,
        'df': df,
        'bank_logo': bank_logo,
        'app_details': app_detail,
        'topic_modeling': topic_modeling,
        'top_negative_reviews_data': top_negative_reviews_data
    }   

    
 

    sentiment_counts = setiment_context(request)
    context['setiment_context'] = {
        'sentiment_counts': sentiment_counts
    }

    sentiment_overtime_data = sentiment_overtime(request)
    context['sentiment_overtime'] = {
        'sentiment_overtime_data': sentiment_overtime_data
    }

    app_version_overtime_data = app_version_overtime(request)
    context['app_version_overtime'] = {
        'app_version_overtime_data': app_version_overtime_data
    }



    print(context)
    return render(request, 'Business-analysis.html', context)



#technical analysis
def format_sgd_million(number):
    # Check if the number is greater than or equal to 1 million
    if abs(number) >= 1_000_000:
        # Format the number in million SGD with two decimal places
        formatted_number = f"{number / 1_000_000:.2f} M"
    else:
        # If the number is less than 1 million, simply format it as SGD with two decimal places
        formatted_number = f"{number:.2f}"
    
    return formatted_number


def filter_data(request):
    start_date = request.POST.get('start_date')
    end_date = request.POST.get('end_date')
    print(start_date)
    print(end_date)

    if start_date and end_date is not None:
        # Perform filtering based on start_date and end_date
        technical_analysis = fetch_bank_reviews_data()[1]

        data = pd.read_csv(technical_analysis.analysis_file)

        data['Date and Time'] = pd.to_datetime(data['Date and Time'])
        filtered_df = (data['Date and Time'].dt.date >= pd.to_datetime(start_date).date()) & (data['Date and Time'].dt.date <= pd.to_datetime(end_date).date())
        filtered_data = data[filtered_df]
        
        
    else:
        # If no dates are selected, show all data
        technical_analysis = fetch_bank_reviews_data()[1]

        data = pd.read_csv(technical_analysis.analysis_file)
        
        
        filtered_data = data
        print(filtered_data)
    return filtered_data


def read_technical_analysis_data(request):
    selected_bank = fetch_selected_bank()
    # technical_analysis = fetch_bank_reviews_data()[1]
    filtered_data = filter_data(request)
    df = filtered_data
    print(df)
    df['Transaction Time'] = pd.to_datetime(df['Transaction Time'])
    
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])  # Convert 'Date and Time' to datetime

    # Calculate the average transaction time
    average_transaction_time = (df['Transaction Time'] - df['Date and Time']).mean().total_seconds()

    # Calculate other statistics
    total_transactions_amount = df['Original Transaction Amount'].sum()
    successful_transactions_amount = df[df['Result'] == 'Success']['Original Transaction Amount'].sum()
    failed_transactions_amount = df[df['Result'] == 'Failure']['Original Transaction Amount'].sum()
    technical_declined_transactions_amount = df[df['Decline Type'] == 'Technical']['Original Transaction Amount'].sum()
    business_declined_transactions_amount = df[df['Decline Type'] == 'Business']['Original Transaction Amount'].sum()
    average_transaction_amount = df['Transaction Amount'].mean()
    average_transaction_time = (df['Transaction Time'] - df['Date and Time']).mean().total_seconds()
    transaction_reversal_amount = df[df['Transaction Reversal'] == 'Yes']['Transaction Amount'].sum()
    hourly_failed_transactions = df[df['Decline Type'] == 'Technical'].groupby(df['Transaction Time'].dt.hour)['Original Transaction Amount'].sum()
    weekly_failed_transactions = df[df['Decline Type'] == 'Technical'].groupby(df['Transaction Time'].dt.strftime("%Y-%m-%d"))['Original Transaction Amount'].sum()
    hourl_buisness_decline_type = df[df['Decline Type'] == 'Business'].groupby(df['Transaction Time'].dt.hour)['Original Transaction Amount'].sum()
    weekly_buisness_decline_type = df[df['Decline Type'] == 'Business'].groupby(df['Transaction Time'].dt.strftime("%Y-%m-%d"))['Original Transaction Amount'].sum()

    df['Hour of Day'] = df['Transaction Time'].dt.hour
    # Calculate technical decline and business decline amounts on an hourly basis
    hourly_technical_declined = df[df['Decline Type'] == 'Technical'].groupby([df['Date and Time'].dt.date, 'Hour of Day'])['Original Transaction Amount'].sum()
    hourly_business_declined = df[df['Decline Type'] == 'Business'].groupby([df['Date and Time'].dt.date, 'Hour of Day'])['Original Transaction Amount'].sum()

    # Filter rows for technical and business categories
    technical_responses = df[df['Decline Type'] == "Technical"]
    business_responses = df[df['Decline Type'] == "Business"]

    # Count occurrences of each response message for technical and business categories
    top_technical_responses = technical_responses['Response Message'].value_counts()[0:5]
    top_business_responses = business_responses['Response Message'].value_counts()[0:5]

    # Create the dataset
    Table_data = [
        {
            'title': 'Top Technical Responses',
            'data': top_technical_responses.to_dict()
        },
        {
            'title': 'Top Business Responses',
            'data': top_business_responses.to_dict()
        }
    ]
    

# Create the chart_data list of dictionaries
    labels = []  # Hour labels
    chart_data = []  # Amounts for each hour

    # Loop through the grouped data and create a dictionary for each hour
    for hour, amount in hourly_failed_transactions.items():
        labels.append(hour)
        chart_data.append(amount)

        chart_data_dict = {
            'labels': labels,
            'chart_data': chart_data
        }


# Create the weekly chart_data list of dictionaries
    week_labels = [] # Hour labels
    week_chart_data = [] # Amounts for each hour

    # Loop through the grouped data and create a dictionary for each hour
    for day, amount in weekly_failed_transactions.items():
        week_labels.append(day)
        week_chart_data.append(amount)

        week_chart_data_dict = {
            'week_labels': week_labels,
            'week_chart_data': week_chart_data
        }

    business_labels = []  # Hour labels
    business_chart_data = []  # Amounts for each hour

    # Loop through the grouped data and create a dictionary for each hour
    for hour, amount in hourl_buisness_decline_type.items():
        business_labels.append(hour)
        business_chart_data.append(amount)

        business_chart_data_dict = {
            'business_labels': business_labels,
            'business_chart_data': business_chart_data
        }

    business_week_labels = [] # Hour labels
    business_week_chart_data = [] # Amounts for each hour

    # Loop through the grouped data and create a dictionary for each hour
    for day, amount in weekly_buisness_decline_type.items():
        business_week_labels.append(day)
        business_week_chart_data.append(amount)

        business_week_chart_data_dict = {
            'business_week_labels': business_week_labels,
            'business_week_chart_data': business_week_chart_data
        }


    heatmap_labels = l = [i for i in range(0,24)]
    # heatmap_data = [] # Heat map data

    # Loop through the grouped data and create a dictionary for each hour with each date
    # for date, hour_amounts in hourly_technical_declined.items():
    #     for hour, amount in hour_amounts.items():
    #         heatmap_labels.append(hour)
    #         # heatmap_data.append(amount)
    
    def convert_data_to_list(hourly_data_dict):
        data_dict = hourly_data_dict.to_dict()
        data_by_date = defaultdict(list)

        # Iterate through the original dictionary and group data by date
        for (date, hour), value in data_dict.items():
            date_str = date.strftime('%Y-%m-%d')
            data_by_date[date_str].append(value)

        # Convert the defaultdict into the desired format
        result_list = [{'name': date, 'data': data} for date, data in data_by_date.items()]

        # Sort the result by date
        result_list.sort(key=lambda x: x['name'])

        return result_list
    
    # Convert the hourly_technical_declined dictionary into the desired format
    td_hour = convert_data_to_list(hourly_technical_declined)
    bd_hour = convert_data_to_list(hourly_business_declined)


    # Create the heatmap_data dictionary
    technical_heatmap_data_dict = {
        'labels': heatmap_labels,
        'data': td_hour
    }

    # Create the heatmap_data dictionary
    business_heatmap_data_dict = {
        'labels': heatmap_labels,
        'data': bd_hour
    }

    

    # # Create a dictionary with the calculated statistics
    

    transaction_data = [{
        "title":"Total Transactions ",
        "value":format_sgd_million(total_transactions_amount),
        "icon":"total.svg"
    },
    {
        "title":"Success Transactions",
        "value":format_sgd_million(successful_transactions_amount),
        "icon":"Success.svg"
    },
    {
        "title":"Failed Transactions",
        "value":format_sgd_million(failed_transactions_amount),
        "icon":"Failure.svg"
    },
    {
        "title":"Technical Decline",
        "value":format_sgd_million(technical_declined_transactions_amount),
        "icon":"Technical Decline.svg"
    },
    {
        "title":"Business Decline",
        "value":format_sgd_million(business_declined_transactions_amount),
        "icon":"Business decline.svg"
    },
    {
        "title":"Average Transaction",
        "value":format_sgd_million(average_transaction_amount),
        "icon":"Average.svg"
    },
    {
        "title":"TRT",
        "value":f"{average_transaction_time:.2f} ms",
        "icon":"TRT.svg"
    },
    {
        "title":"Transaction Reversal",
        "value":format_sgd_million(transaction_reversal_amount),
        "icon":"reversal.svg"
    }

    ]
    
    return transaction_data, chart_data_dict , week_chart_data_dict, business_chart_data_dict, business_week_chart_data_dict, technical_heatmap_data_dict, business_heatmap_data_dict, Table_data

import random
from datetime import datetime, timedelta

def generate_dates(base_date, num_days):
    date_list = [base_date + timedelta(days=i) for i in range(num_days)]
    return date_list

def heatmap_view(request):
    # Generate random data for demonstration
    data = [
        [random.randint(0, 100) for _ in range(24)] for _ in range(7)
    ]
    base_date = datetime(2023, 9, 20)  # Your base date
    num_days = 7  # Number of days

    dates = generate_dates(base_date, num_days)
    hours = list(range(24))

    # Restructure the data as a list of dictionaries
    heatmap_data = []

    for date, values in zip(dates, data):
        day_data = {
            'date': date.strftime('%d-%m-%Y'),
            'values': values
        }
        heatmap_data.append(day_data)

    context = {
        'dates': dates,  # Pass the list of dates
        'hours': hours,
        'data': heatmap_data  # Pass the restructured data
    }
    return context







@login_required(login_url='LoginPage')
def Technical_analysis(request):
    selected_bank = fetch_selected_bank()
    bank_logo = selected_bank.logo.url
    heatmap_data = heatmap_view(request)
    transaction = read_technical_analysis_data(request)
    chart_data = transaction[1]
    week_data = transaction[2]
    business_data = transaction[3]
    business_data_week_data = transaction[4]
    heatmap_data_td = transaction[5]
    heatmap_data_bd = transaction[6]
    Table_data = transaction[7]
    
    context = {
        'heatmap_data': heatmap_data,
        'transaction': transaction[0],
        'bank': selected_bank,
        'bank_logo': bank_logo,
         
    }


    context['chart_data'] = {
        'labels': chart_data['labels'],
        'chart_data': chart_data['chart_data']
    }

    context['week_chart_data'] = {
        'week_labels': week_data['week_labels'],
        'week_chart_data': week_data['week_chart_data']
    }

    context['business_chart_data'] = {
        'business_labels': business_data['business_labels'],
        'business_chart_data': business_data['business_chart_data']
    }

    context['business_week_chart_data'] = {
        'business_week_labels': business_data_week_data['business_week_labels'],
        'business_week_chart_data': business_data_week_data['business_week_chart_data']
    }

    context['technical_heatmap_data'] = {
        'labels': heatmap_data_td['labels'],
        'data': heatmap_data_td['data']
    }

    context['business_heatmap_data'] = {
        'labels': heatmap_data_bd['labels'],
        'data': heatmap_data_bd['data']
    }

    context['Table_data'] = Table_data

    
    return render(request,'Technical-analysis.html',context)


def Report_analysis(request):
    selected_bank = fetch_selected_bank()
    bank_logo = selected_bank.logo.url
    context = {
        'bank': selected_bank,
        'bank_logo': bank_logo
    }
    return render(request,'Report.html',context)


def Technical_data(request):
    selected_bank = fetch_selected_bank()
    technical_analysis = fetch_bank_reviews_data()[1]
    df = pd.read_csv(technical_analysis.analysis_file)[:5]
    context = {
        'bank': selected_bank,
        'df': df
    }
    return render(request,'Technical-data.html',context)




def heatmap(df):
    df['Transaction Time'] = pd.to_datetime(df['Transaction Time'], utc=True)
    df['Date and Time'] = pd.to_datetime(df['Date and Time'], utc=True)
    df['Hour of Day'] = df['Transaction Time'].dt.hour

    # Calculate technical decline and business decline amounts on an hourly basis
    hourly_technical_declined = df[df['Decline Type'] == 'Technical'].groupby([df['Date and Time'].dt.date, 'Hour of Day'])['Original Transaction Amount'].sum()
    hourly_business_declined = df[df['Decline Type'] == 'Business'].groupby([df['Date and Time'].dt.date, 'Hour of Day'])['Original Transaction Amount'].sum()

    # create a dictionary with two keys: labels and data
    heatmap_data = {
        'labels': [i for i in range(0,24)],
        'data': []
    }

    # Loop through the grouped data and create a dictionary for each hour with each date
    for date, hour_amounts in hourly_technical_declined.items():
        for hour, amount in hour_amounts.items():
            heatmap_data['data'].append({
                'name': date,
                'data': amount
            })

    # Sort the data by date
    heatmap_data['data'].sort(key=lambda x: x['name'])

    return heatmap_data

