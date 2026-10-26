# AI-Based Student Performance Prediction and Personalized Recommendation System

This project is a beginner-friendly full-stack application that uses a Random Forest classifier to predict a student's performance level based on academic and behavioral features. It also generates personalized improvement suggestions and stores prediction history in SQLite.

## Project Overview

The system helps students and faculty understand likely academic performance by analyzing inputs such as attendance, internal marks, assignment scores, previous semester marks, study hours, and completed assignments. The app trains a machine learning model on a sample dataset and makes predictions in real time through a Flask web application.

## Features

- Student performance prediction using Random Forest Classifier
- Personalized recommendations for each student
- Confidence percentage for every prediction
- Prediction history stored in SQLite database
- Dashboard with total predictions and class-wise counts
- Responsive and modern UI built using HTML, CSS, and JavaScript
- Input validation and error handling
- Beginner-friendly code structure

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- SQLite
- HTML
- CSS
- JavaScript

## Machine Learning Algorithm

The project uses the Random Forest Classifier from Scikit-learn.

Why Random Forest?

- Handles moderate-sized tabular data well
- Produces good accuracy for classification tasks
- Easy to explain in interviews and projects
- Works efficiently with numeric student features

## Project Structure

```text
student-performance-ai/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── student_performance.db
├── dataset/
│   └── student_data.csv
├── model/
│   └── student_model.pkl
├── templates/
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── venv/
```

## Installation Steps

1. Open a terminal in the project folder.
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

- Windows:

```bash
venv\Scripts\activate
```

- Linux/macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Train the Model

Run the following command:

```bash
python train_model.py
```

This command loads the dataset from the dataset folder, trains the Random Forest model, and saves the trained model in the model folder as student_model.pkl.

## How to Run the Flask Application

Start the app with:

```bash
python app.py
```

Then open the browser and visit:

```text
http://127.0.0.1:5000/
```

## Example Usage

1. Enter the student's name.
2. Provide attendance, internal marks, assignment score, previous semester marks, study hours, and completed assignments.
3. Click the Predict Performance button.
4. View the predicted class, confidence percentage, and tailored recommendations.
5. Visit the dashboard page to check all stored predictions.

## Future Enhancements

- Add login and role-based access for student/faculty panels
- Use a larger dataset with more student records
- Add visual charts for performance analysis
- Integrate with a real student management system
- Add model accuracy improvement with hyperparameter tuning
- Deploy to cloud platforms like Render or Railway

## Notes

This project is intentionally designed to be easy to understand and explain during technical interviews. The code uses simple Flask routes, a straightforward dataset, and a clean machine learning pipeline so that the overall flow is easy to follow.
