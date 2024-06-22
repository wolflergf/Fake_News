# Fake News Detection Project

## Overview

This project implements a machine learning model to detect fake news articles. It includes a Python-based model training script and a Flask web application for real-time predictions.

## Features

- Machine learning model to classify news articles as real or fake
- Flask web application for easy interaction with the model
- TF-IDF vectorization for text feature extraction
- Logistic Regression classifier
- Model persistence for quick loading

## Technologies Used

- Python 3.x
- Flask
- scikit-learn
- pandas
- joblib

## Project Structure

- `main.py`: Script for data preprocessing, model training, and evaluation
- `app.py`: Flask application for serving predictions
- `fake_news_model.pkl`: Trained model file
- `vectorizer.pkl`: TF-IDF vectorizer
- `templates/index.html`: HTML template for the web interface (not included in the provided code)

## Installation

1. Clone this repository:
git clone https://github.com/wolflergf/fake-news-detection.git
cd fake-news-detection

2. Install the required packages:
pip install -r requirements.txt

## Usage

### Training the Model

1. Ensure you have a CSV file named `news_articles.csv` with columns 'text' and 'label' in the project directory.
2. Run the training script:
python main.py

3. The script will output model performance metrics and save the trained model and vectorizer.

### Running the Web Application

1. Start the Flask server:
python app.py

2. Open a web browser and navigate to `http://localhost:5000`.
3. Enter a news article text in the provided interface to get a prediction.

## Model Performance

The current model achieves the following performance (example metrics):

- Accuracy: 0.92
- Precision: 0.91
- Recall: 0.93
- F1-score: 0.92

## Future Improvements

- Implement more advanced NLP techniques (e.g., word embeddings, BERT)
- Add data augmentation to improve model robustness
- Incorporate feature importance analysis
- Implement user feedback loop for continuous model improvement

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Wolfler Guzzo Ferreira - [wolflergf@gmail.com](mailto:wolflergf@gmail.com)

Project Link: [https://github.com/wolflergf/fake-news-detection](https://github.com/wolflergf/fake-news-detection)