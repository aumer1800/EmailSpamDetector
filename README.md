# # SMS Spam Detection

This project builds a machine learning model to classify SMS messages as spam or ham (legitimate). It uses the SMS Spam Collection dataset from Kaggle, applies text preprocessing, and compares two classification models to find the best performer for spam detection.

## What I Did

### 1. Data Collection
- Uploaded my Kaggle API credentials (kaggle.json) to authenticate with Kaggle.
- Downloaded the "SMS Spam Collection" dataset directly from Kaggle using the Kaggle API.
- Extracted the dataset from the downloaded zip file.

### 2. Data Cleaning
- Loaded the dataset into a pandas DataFrame using latin-1 encoding.
- Removed unnecessary extra columns, keeping only the label and message columns.
- Renamed the columns to `label` and `message` for clarity.
- Checked the dataset shape and looked for missing values.
- Checked for and removed duplicate rows.
- Reviewed the class distribution (spam vs ham) and calculated the percentage of each class.
- Checked for any empty messages in the dataset.

### 3. Exploratory Data Analysis
- Plotted a bar chart comparing the number of spam messages against ham messages.
- Reviewed value counts and unique label values to confirm the dataset only contains two classes.

### 4. Text Preprocessing
- Wrote a custom text cleaning function to prepare the messages for the TF-IDF vectorizer.
- The cleaning function does the following:
  - Converts all text to lowercase.
  - Expands contractions (for example, changing "don't" to "do not") using the contractions library.
  - Removes URLs from the messages.
  - Removes punctuation and numbers, keeping only letters and spaces.
  - Removes extra whitespace.
  - Removes common English stopwords using NLTK's stopwords list.
- Applied this cleaning function to every message and stored the result in a new column called `clean_message`.

### 5. Train/Test Split
- Split the cleaned data into training and testing sets (80 percent train, 20 percent test).
- Used stratified sampling based on the label column so that both sets keep the same spam-to-ham ratio.

### 6. Feature Extraction
- Used a TF-IDF Vectorizer to convert the cleaned text messages into numerical features.
- Fit the vectorizer only on the training data to avoid data leakage, then used it to transform both the training and test data.

### 7. Model 1: Multinomial Naive Bayes
- Trained a Multinomial Naive Bayes classifier on the TF-IDF features.
- Generated predictions on the test set.
- Evaluated the model using accuracy, precision, recall, and F1-score, along with a full classification report.
- Built and visualized a confusion matrix for this model.

### 8. Model 2: Linear SVM
- Trained a Linear Support Vector Machine (LinearSVC) classifier on the same TF-IDF features.
- Generated predictions on the test set.
- Evaluated the model using the same set of metrics: accuracy, precision, recall, and F1-score.
- Built and visualized a confusion matrix for this model.

### 9. Model Comparison
- Created a comparison table summarizing the accuracy, precision, recall, and F1-score of both models side by side.
- Analyzed why recall matters more than accuracy alone in spam detection, since a false negative (spam getting into the inbox) is a bigger practical problem than a false positive in many cases.
- Compared the recall scores of both models and found that Linear SVM achieved a higher spam recall (83.21 percent) than Multinomial Naive Bayes (71.76 percent), while still maintaining strong precision (95.61 percent). Based on this, Linear SVM was chosen as the preferred model.

### 10. Bonus: WordCloud Visualization
- Generated a WordCloud for spam messages to visualize the most frequently used words in spam.
- Generated a separate WordCloud for ham messages to visualize the most frequently used words in legitimate messages.
- Compared the two WordClouds and noted that spam messages tend to use promotional, prize, and urgency-related language, while ham messages use more conversational and personal language.
- Noted that WordClouds are useful for visual intuition about word frequency but do not directly show which words are most important for classification, since that role is filled by the TF-IDF features used in the models.

## Tools and Libraries Used
- pandas for data loading and manipulation
- matplotlib for plotting and visualization
- NLTK for stopword removal
- contractions for expanding contracted words
- scikit-learn for train/test splitting, TF-IDF vectorization, model training, and evaluation
- wordcloud for generating word cloud visualizations

## Dataset
The dataset used is the SMS Spam Collection Dataset from Kaggle (uciml/sms-spam-collection-dataset), which contains a set of SMS messages labeled as either spam or ham.

## Results Summary
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Multinomial Naive Bayes | See notebook output | See notebook output | 71.76% | See notebook output |
| Linear SVM | See notebook output | 95.61% | 83.21% | See notebook output |

Linear SVM was selected as the better model overall because it detects a larger share of actual spam messages (higher recall) while keeping precision high.