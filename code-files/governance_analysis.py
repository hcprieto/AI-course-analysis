"""
Governance & Rights Analysis for Global Thriving Study

This module performs comprehensive text analysis of UN General Assembly speeches to extract:
- Governance themes and patterns
- Democratic values and rights mentions
- Sentiment analysis by country and region
- Topic modeling and keyword extraction
- Cross-country comparisons of governance discourse

Author: AI Analysis System
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter, defaultdict
from pathlib import Path
import warnings
from datetime import datetime
import base64
from io import BytesIO

# Text processing libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class GovernanceAnalyzer:
    def __init__(self, df):
        self.df = df
        self.governance_df = None
        self.results = {}
        self.charts = {}
        
        # Initialize text processing tools
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define governance-related keywords
        self.governance_keywords = {
            'democracy': ['democracy', 'democratic', 'democratization', 'election', 'voting', 'ballot', 'parliament', 'representative'],
            'human_rights': ['human rights', 'civil rights', 'freedom', 'liberty', 'equality', 'discrimination', 'justice', 'fairness'],
            'rule_of_law': ['rule of law', 'legal', 'constitution', 'judicial', 'court', 'law enforcement', 'accountability', 'transparency'],
            'governance': ['governance', 'government', 'administration', 'leadership', 'management', 'oversight', 'institution'],
            'corruption': ['corruption', 'corrupt', 'bribery', 'fraud', 'misconduct', 'ethics', 'integrity', 'clean government'],
            'civil_society': ['civil society', 'ngo', 'activist', 'citizen', 'community', 'participation', 'engagement', 'advocacy'],
            'press_freedom': ['press freedom', 'media', 'journalism', 'censorship', 'information', 'expression', 'speech'],
            'women_rights': ['women rights', 'gender equality', 'women empowerment', 'gender discrimination', 'female participation']
        }
        
    def prepare_data(self):
        """Prepare and clean text data for analysis"""
        print("Preparing governance text data...")
        
        # Create working copy
        self.governance_df = self.df.copy()
        
        # Filter out rows without proper text data
        valid_text_mask = (
            self.governance_df['Text'].notna() & 
            (self.governance_df['Text'] != 'Not Avalable') &
            (self.governance_df['Text'].str.len() > 100)  # Minimum length threshold
        )
        
        self.governance_df = self.governance_df[valid_text_mask].copy()
        
        # Clean text data
        self.governance_df['clean_text'] = self.governance_df['Text'].apply(self._clean_text)
        self.governance_df['word_count'] = self.governance_df['clean_text'].apply(lambda x: len(x.split()))
        
        print(f"Text data prepared: {len(self.governance_df)} countries with valid speeches")
        print(f"Average speech length: {self.governance_df['word_count'].mean():.0f} words")
        
        return self.governance_df
    
    def _clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def sentiment_analysis(self):
        """Perform sentiment analysis on UN speeches"""
        print("Performing sentiment analysis...")
        
        # Calculate sentiment scores
        sentiment_scores = []
        for text in self.governance_df['Text']:
            scores = self.sia.polarity_scores(text)
            sentiment_scores.append(scores)
        
        # Create sentiment dataframe
        sentiment_df = pd.DataFrame(sentiment_scores)
        
        # Add to main dataframe
        self.governance_df['sentiment_positive'] = sentiment_df['pos']
        self.governance_df['sentiment_negative'] = sentiment_df['neg']
        self.governance_df['sentiment_neutral'] = sentiment_df['neu']
        self.governance_df['sentiment_compound'] = sentiment_df['compound']
        
        # Categorize sentiment
        def categorize_sentiment(score):
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.governance_df['sentiment_category'] = self.governance_df['sentiment_compound'].apply(categorize_sentiment)
        
        # Calculate regional sentiment statistics
        regional_sentiment = self.governance_df.groupby('Continent').agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        }).round(3)
        
        self.results['sentiment_analysis'] = {
            'overall_sentiment': {
                'mean_compound': float(self.governance_df['sentiment_compound'].mean()),
                'positive_speeches': int((self.governance_df['sentiment_compound'] > 0.05).sum()),
                'negative_speeches': int((self.governance_df['sentiment_compound'] < -0.05).sum()),
                'neutral_speeches': int(((self.governance_df['sentiment_compound'] >= -0.05) & (self.governance_df['sentiment_compound'] <= 0.05)).sum())
            },
            'regional_sentiment': regional_sentiment.to_dict(),
            'country_sentiment': self.governance_df[['Country', 'Continent', 'sentiment_compound', 'sentiment_category']].to_dict('records')
        }
        
        return self.results['sentiment_analysis']
    
    def governance_themes_analysis(self):
        """Analyze governance themes in speeches"""
        print("Analyzing governance themes...")
        
        # Initialize theme tracking
        theme_counts = defaultdict(lambda: defaultdict(int))
        country_themes = []
        
        for idx, row in self.governance_df.iterrows():
            country = row['Country']
            continent = row['Continent']
            text = row['clean_text']
            
            # Count theme keywords in text
            country_theme_score = {'Country': country, 'Continent': continent}
            
            for theme, keywords in self.governance_keywords.items():
                count = 0
                for keyword in keywords:
                    # Count occurrences (case insensitive)
                    count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
                
                country_theme_score[theme] = count
                theme_counts[theme][country] = count
            
            country_themes.append(country_theme_score)
        
        # Create themes dataframe
        themes_df = pd.DataFrame(country_themes)
        
        # Calculate theme statistics
        theme_stats = {}
        for theme in self.governance_keywords.keys():
            theme_stats[theme] = {
                'total_mentions': int(themes_df[theme].sum()),
                'countries_mentioning': int((themes_df[theme] > 0).sum()),
                'avg_mentions_per_country': float(themes_df[theme].mean()),
                'max_mentions': int(themes_df[theme].max()),
                'top_countries': themes_df.nlargest(5, theme)[['Country', theme]].to_dict('records')
            }
        
        # Regional theme analysis
        regional_themes = themes_df.groupby('Continent')[list(self.governance_keywords.keys())].sum()
        
        self.results['governance_themes'] = {
            'theme_statistics': theme_stats,
            'regional_themes': regional_themes.to_dict(),
            'country_themes': themes_df.to_dict('records')
        }
        
        # Add themes to main dataframe
        for theme in self.governance_keywords.keys():
            self.governance_df[f'theme_{theme}'] = themes_df[theme]
        
        return self.results['governance_themes']
    
    def topic_modeling(self, n_topics=5):
        """Perform topic modeling on speeches"""
        print(f"Performing topic modeling with {n_topics} topics...")
        
        # Prepare text for topic modeling
        documents = self.governance_df['clean_text'].tolist()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Fit and transform documents
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        # Perform LDA topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': [float(topic[i]) for i in top_words_idx]
            })
        
        # Get document-topic probabilities
        doc_topic_probs = lda.transform(doc_term_matrix)
        
        # Assign dominant topic to each document
        self.governance_df['dominant_topic'] = doc_topic_probs.argmax(axis=1)
        self.governance_df['topic_probability'] = doc_topic_probs.max(axis=1)
        
        self.results['topic_modeling'] = {
            'topics': topics,
            'document_topics': self.governance_df[['Country', 'dominant_topic', 'topic_probability']].to_dict('records')
        }
        
        return self.results['topic_modeling']
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating governance visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Sentiment Analysis by Region
        self._create_sentiment_visualizations()
        
        # 2. Governance Themes Analysis
        self._create_themes_visualizations()
        
        # 3. Word Cloud
        self._create_wordcloud()
        
        # 4. Topic Distribution
        self._create_topic_visualizations()
        
        return self.charts
    
    def _create_sentiment_visualizations(self):
        """Create sentiment analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('UN Speech Sentiment Analysis by Region', fontsize=16, fontweight='bold')
        
        # 1. Regional sentiment box plot
        sns.boxplot(data=self.governance_df, x='Continent', y='sentiment_compound', ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Distribution by Continent')
        axes[0, 0].set_xlabel('Continent')
        axes[0, 0].set_ylabel('Compound Sentiment Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sentiment category counts
        sentiment_counts = self.governance_df['sentiment_category'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('Overall Sentiment Distribution')
        
        # 3. Regional sentiment means
        regional_means = self.governance_df.groupby('Continent')['sentiment_compound'].mean().sort_values(ascending=True)
        bars = axes[1, 0].barh(regional_means.index, regional_means.values)
        axes[1, 0].set_title('Average Sentiment by Region')
        axes[1, 0].set_xlabel('Mean Compound Sentiment Score')
        
        # Color bars based on sentiment
        for i, bar in enumerate(bars):
            value = regional_means.values[i]
            if value > 0.05:
                bar.set_color('#2ecc71')  # Green for positive
            elif value < -0.05:
                bar.set_color('#e74c3c')  # Red for negative
            else:
                bar.set_color('#f39c12')  # Orange for neutral
        
        # 4. Sentiment vs Speech Length
        axes[1, 1].scatter(self.governance_df['word_count'], self.governance_df['sentiment_compound'], alpha=0.6)
        axes[1, 1].set_title('Sentiment vs Speech Length')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Compound Sentiment Score')
        
        # Add trend line
        z = np.polyfit(self.governance_df['word_count'], self.governance_df['sentiment_compound'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(self.governance_df['word_count'], p(self.governance_df['word_count']), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        # Save chart
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        self.charts['sentiment_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
    
    def _create_themes_visualizations(self):
        """Create governance themes visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Governance Themes in UN Speeches', fontsize=16, fontweight='bold')
        
        # 1. Theme frequency across all speeches
        theme_totals = {}
        for theme in self.governance_keywords.keys():
            theme_totals[theme] = self.governance_df[f'theme_{theme}'].sum()
        
        sorted_themes = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)
        themes, counts = zip(*sorted_themes)
        
        bars = axes[0, 0].bar(range(len(themes)), counts)
        axes[0, 0].set_title('Total Theme Mentions Across All Speeches')
        axes[0, 0].set_xlabel('Governance Themes')
        axes[0, 0].set_ylabel('Total Mentions')
        axes[0, 0].set_xticks(range(len(themes)))
        axes[0, 0].set_xticklabels([t.replace('_', ' ').title() for t in themes], rotation=45, ha='right')
        
        # Color bars
        colors = sns.color_palette("husl", len(themes))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 2. Regional theme distribution (heatmap)
        regional_themes = self.governance_df.groupby('Continent')[[f'theme_{theme}' for theme in self.governance_keywords.keys()]].mean()
        regional_themes.columns = [col.replace('theme_', '').replace('_', ' ').title() for col in regional_themes.columns]
        
        sns.heatmap(regional_themes.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 1])
        axes[0, 1].set_title('Average Theme Mentions by Region')
        axes[0, 1].set_xlabel('Continent')
        axes[0, 1].set_ylabel('Governance Themes')
        
        # 3. Countries with highest democracy mentions
        top_democracy = self.governance_df.nlargest(10, 'theme_democracy')[['Country', 'theme_democracy']]
        axes[1, 0].barh(top_democracy['Country'], top_democracy['theme_democracy'])
        axes[1, 0].set_title('Top 10 Countries: Democracy Mentions')
        axes[1, 0].set_xlabel('Democracy-related Mentions')
        
        # 4. Human rights vs Democracy mentions scatter
        axes[1, 1].scatter(self.governance_df['theme_democracy'], self.governance_df['theme_human_rights'], alpha=0.6)
        axes[1, 1].set_title('Democracy vs Human Rights Mentions')
        axes[1, 1].set_xlabel('Democracy Mentions')
        axes[1, 1].set_ylabel('Human Rights Mentions')
        
        # Add country labels for high values
        for idx, row in self.governance_df.iterrows():
            if row['theme_democracy'] > 5 or row['theme_human_rights'] > 5:
                axes[1, 1].annotate(row['Country'], 
                                  (row['theme_democracy'], row['theme_human_rights']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        self.charts['themes_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
    
    def _create_wordcloud(self):
        """Create word cloud from all speeches"""
        # Combine all speech text
        all_text = ' '.join(self.governance_df['clean_text'])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in UN Speeches', fontsize=16, fontweight='bold', pad=20)
        
        # Save chart
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        self.charts['wordcloud'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
    
    def _create_topic_visualizations(self):
        """Create topic modeling visualizations"""
        if 'topic_modeling' not in self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Topic Modeling Results', fontsize=16, fontweight='bold')
        
        # 1. Topic distribution
        topic_counts = self.governance_df['dominant_topic'].value_counts().sort_index()
        axes[0].bar(topic_counts.index, topic_counts.values)
        axes[0].set_title('Document Distribution Across Topics')
        axes[0].set_xlabel('Topic ID')
        axes[0].set_ylabel('Number of Documents')
        
        # 2. Regional topic distribution
        topic_by_region = pd.crosstab(self.governance_df['Continent'], self.governance_df['dominant_topic'])
        topic_by_region_pct = topic_by_region.div(topic_by_region.sum(axis=1), axis=0) * 100
        
        sns.heatmap(topic_by_region_pct, annot=True, fmt='.1f', cmap='Blues', ax=axes[1])
        axes[1].set_title('Topic Distribution by Region (%)')
        axes[1].set_xlabel('Topic ID')
        axes[1].set_ylabel('Continent')
        
        plt.tight_layout()
        
        # Save chart
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        self.charts['topic_modeling'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("Generating governance analysis summary...")
        
        summary = {
            'dataset_overview': {
                'total_countries': len(self.governance_df),
                'total_words': int(self.governance_df['word_count'].sum()),
                'avg_speech_length': float(self.governance_df['word_count'].mean()),
                'continents_covered': list(self.governance_df['Continent'].unique())
            },
            'sentiment_overview': {
                'overall_sentiment': float(self.governance_df['sentiment_compound'].mean()),
                'most_positive_country': self.governance_df.loc[self.governance_df['sentiment_compound'].idxmax(), 'Country'],
                'most_negative_country': self.governance_df.loc[self.governance_df['sentiment_compound'].idxmin(), 'Country'],
                'sentiment_range': float(self.governance_df['sentiment_compound'].max() - self.governance_df['sentiment_compound'].min())
            },
            'themes_overview': {
                'most_discussed_theme': max(self.governance_keywords.keys(), 
                                          key=lambda x: self.governance_df[f'theme_{x}'].sum()),
                'total_governance_mentions': sum(self.governance_df[f'theme_{theme}'].sum() 
                                               for theme in self.governance_keywords.keys())
            }
        }
        
        self.results['summary'] = summary
        return summary
    
    def run_complete_analysis(self):
        """Run the complete governance analysis pipeline"""
        print("=== Starting Governance & Rights Analysis ===")
        
        # Prepare data
        self.prepare_data()
        
        # Run all analyses
        self.sentiment_analysis()
        self.governance_themes_analysis()
        self.topic_modeling()
        self.create_visualizations()
        self.generate_summary_statistics()
        
        print("=== Governance analysis completed successfully ===")
        return self.results, self.charts

# Example usage and testing
if __name__ == "__main__":
    # Load the data
    data_path = Path(__file__).parent.parent / "source-files" / "World Data 2.0-Data.csv"
    df = pd.read_csv(data_path)
    
    # Run analysis
    analyzer = GovernanceAnalyzer(df)
    results, charts = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n=== GOVERNANCE ANALYSIS SUMMARY ===")
    print(f"Countries analyzed: {results['summary']['dataset_overview']['total_countries']}")
    print(f"Total words processed: {results['summary']['dataset_overview']['total_words']:,}")
    print(f"Average sentiment: {results['summary']['sentiment_overview']['overall_sentiment']:.3f}")
    print(f"Most discussed theme: {results['summary']['themes_overview']['most_discussed_theme']}")