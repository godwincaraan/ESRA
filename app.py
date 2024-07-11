from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from model import classify_cleaned, classify_sent, classify_rating
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file_input' not in request.files:
        return "No file part"
    
    file = request.files['file_input']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
   
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Initialize empty lists to store classification results
        classification_results = []
        ratings = []
        sentiments = []
        emotions = []

        # Apply classification functions to the text in the first column
        for index, row in df.iterrows():
            # Call the classification functions and store the results
            emotion_result = classify_cleaned(row.iloc[0])
            sentiment_result = classify_sent(row.iloc[0])
            rating_result = classify_rating(row.iloc[0])
            
            # Append the results to their respective lists
            emotions.append(emotion_result)
            sentiments.append(sentiment_result)
            ratings.append(rating_result)

        # Create a DataFrame to store the classification results
        classification_df = pd.DataFrame({
            'Text': df.iloc[:, 0],  
            'Emotion': emotions,
            'Sentiment': sentiments,
            'Rating': ratings,
            'Product': df.iloc[:, 1]  
        })

        # Calculate total order count
        total_order_count = len(classification_df)

        # Convert numpy arrays in into strings
        classification_df['Rating'] = classification_df['Rating'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else x)
        classification_df['Emotion'] = classification_df['Emotion'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else x)
        classification_df['Sentiment'] = classification_df['Sentiment'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else x)

        # Create pie chart for emotions
        plt.figure(figsize=(3, 3))  
        emotion_counts = classification_df['Emotion'].value_counts()
        emotion_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Emotion Distribution')
        plt.axis('equal')
        plt.ylabel('')
        emotion_pie = plot_to_base64_string()

        # Create pie chart for sentiments
        plt.figure(figsize=(3, 3)) 
        sentiment_counts = classification_df['Sentiment'].value_counts()
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Sentiment Distribution')
        plt.axis('equal')
        plt.ylabel('')
        sentiment_pie = plot_to_base64_string()

        # Create pie chart for ratings
        plt.figure(figsize=(3, 3)) 
        rating_counts = classification_df['Rating'].value_counts()
        rating_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Rating Distribution')
        plt.axis('equal')
        plt.ylabel('')
        rating_pie = plot_to_base64_string()


        # Apply group by
        rating_counts_by_product = classification_df.groupby('Product')['Rating'].value_counts()
        sentiment_counts_by_product = classification_df.groupby('Product')['Sentiment'].value_counts()
        emotion_counts_by_product = classification_df.groupby('Product')['Emotion'].value_counts()

        # Create bar chart for sentiments
        plt.figure(figsize=(2, 2))  
        sentiment_counts_by_product.unstack().plot(kind='bar', stacked=True) 
        plt.title('Sentiment Distribution per Product')
        plt.xlabel('Product')
        plt.xticks(rotation=360)
        plt.ylabel('')       
        plt.grid(False)
        sentiment_bar = plot_to_base64_string()

        # Create bar chart for emotions
        plt.figure(figsize=(2, 2))   
        emotion_counts_by_product.unstack().plot(kind='bar', stacked=True) 
        plt.title('Emotion Distribution per Product')
        plt.xlabel('Product')
        plt.xticks(rotation=360)
        plt.ylabel('')
        plt.grid(False)
        emotion_bar = plot_to_base64_string()

        # Create bar chart for rating
        plt.figure(figsize=(2, 2))  
        rating_counts_by_product.unstack().plot(kind='bar', stacked=True) 
        plt.title('Rating Distribution per Product')
        plt.xlabel('Product')
        plt.ylabel('')
        plt.xticks(rotation=360)
        plt.grid(False)
        rating_bar = plot_to_base64_string()


        return render_template('index.html', 
                       classification_results=classification_results, 
                       rating_pie=rating_pie, 
                       sentiment_pie=sentiment_pie, 
                       emotion_pie=emotion_pie, 
                       rating_bar=rating_bar, 
                       sentiment_bar=sentiment_bar, 
                       emotion_bar=emotion_bar,
                       total_order_count=total_order_count,
                       classification_df=classification_df.to_html(classes=['table', 'neumorphic']))


def plot_to_base64_string():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    return graphic

if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True)
