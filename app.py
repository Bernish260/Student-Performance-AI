import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_model.pkl"
DB_PATH = BASE_DIR / "student_performance.db"
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_mark",
    "study_hours",
    "completed_assignments",
]

app = Flask(__name__)
app.config["SECRET_KEY"] = "ai-student-performance-project"


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db_connection()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            attendance REAL NOT NULL,
            internal_marks REAL NOT NULL,
            assignment_score REAL NOT NULL,
            previous_mark REAL NOT NULL,
            study_hours REAL NOT NULL,
            completed_assignments INTEGER NOT NULL,
            predicted_performance TEXT NOT NULL,
            confidence REAL NOT NULL,
            prediction_time TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()


init_db()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please train the model first by running: python train_model.py"
        )

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def validate_form_data(form_data):
    errors = []

    student_name = form_data.get("student_name", "").strip()
    if not student_name:
        errors.append("Student name is required.")

    try:
        attendance = float(form_data.get("attendance", ""))
        if attendance < 0 or attendance > 100:
            errors.append("Attendance must be between 0 and 100.")
    except ValueError:
        errors.append("Attendance must be a valid number.")

    for field_name in [
        "internal_marks",
        "assignment_score",
        "previous_mark",
    ]:
        try:
            value = float(form_data.get(field_name, ""))
            if value < 0 or value > 100:
                errors.append(f"{field_name.replace('_', ' ').title()} must be between 0 and 100.")
        except ValueError:
            errors.append(f"{field_name.replace('_', ' ').title()} must be a valid number.")

    try:
        study_hours = float(form_data.get("study_hours", ""))
        if study_hours < 0 or study_hours > 24:
            errors.append("Study hours must be between 0 and 24.")
    except ValueError:
        errors.append("Study hours must be a valid number.")

    try:
        completed_assignments = int(form_data.get("completed_assignments", ""))
        if completed_assignments < 0:
            errors.append("Completed assignments cannot be negative.")
    except ValueError:
        errors.append("Completed assignments must be a valid integer.")

    return errors


def generate_recommendations(prediction, form_data):
    recommendations = []

    if prediction == "High":
        recommendations.append("Keep your current study routine and continue improving with advanced practice.")
        recommendations.append("Maintain consistent attendance and revise difficult concepts once every week.")
        recommendations.append("Challenge yourself with extra projects and leadership activities to stay ahead.")
        return recommendations

    if prediction == "Medium":
        recommendations.append("Increase daily study hours by at least 1 to 2 hours for better consistency.")
        recommendations.append("Focus on weak topics in assignments and internal tests before the next assessment.")
        recommendations.append("Work on time management and complete assignments before deadlines.")
    else:
        recommendations.append("Improve attendance gradually and aim to attend all class sessions consistently.")
        recommendations.append("Set a fixed study schedule and spend more time on revision and practice questions.")
        recommendations.append("Complete pending assignments early and ask for help on difficult concepts from faculty or peers.")

    if float(form_data["assignment_score"]) < 70:
        recommendations.append("Practice assignment-based tasks regularly to strengthen practical understanding.")

    if float(form_data["internal_marks"]) < 60:
        recommendations.append("Revise internal test topics shortly after each lecture and solve sample questions.")

    if float(form_data["study_hours"]) < 3:
        recommendations.append("Create a focused daily study plan to build discipline and improve academic performance.")

    return recommendations[:5]


def save_prediction(student_name, form_data, predicted_label, confidence):
    connection = get_db_connection()
    connection.execute(
        """
        INSERT INTO predictions (
            student_name,
            attendance,
            internal_marks,
            assignment_score,
            previous_mark,
            study_hours,
            completed_assignments,
            predicted_performance,
            confidence,
            prediction_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student_name,
            float(form_data["attendance"]),
            float(form_data["internal_marks"]),
            float(form_data["assignment_score"]),
            float(form_data["previous_mark"]),
            float(form_data["study_hours"]),
            int(form_data["completed_assignments"]),
            predicted_label,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    connection.commit()
    prediction_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
    connection.close()
    return prediction_id


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {key: value for key, value in request.form.items()}
    errors = validate_form_data(form_data)

    if errors:
        return render_template("index.html", errors=errors, form_data=form_data)

    feature_vector = np.array(
        [
            [
                float(form_data["attendance"]),
                float(form_data["internal_marks"]),
                float(form_data["assignment_score"]),
                float(form_data["previous_mark"]),
                float(form_data["study_hours"]),
                float(form_data["completed_assignments"]),
            ]
        ],
        dtype=float,
    )

    model = load_model()
    predicted_label = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    confidence = float(max(probabilities)) * 100
    confidence = round(confidence, 2)

    prediction_id = save_prediction(
        form_data["student_name"].strip(),
        form_data,
        predicted_label,
        confidence,
    )

    prediction_record = {
        "id": prediction_id,
        "student_name": form_data["student_name"].strip(),
        "attendance": float(form_data["attendance"]),
        "internal_marks": float(form_data["internal_marks"]),
        "assignment_score": float(form_data["assignment_score"]),
        "previous_mark": float(form_data["previous_mark"]),
        "study_hours": float(form_data["study_hours"]),
        "completed_assignments": int(form_data["completed_assignments"]),
        "performance": predicted_label,
        "confidence": confidence,
    }

    recommendations = generate_recommendations(predicted_label, form_data)
    return render_template("result.html", prediction=prediction_record, recommendations=recommendations)


@app.route("/dashboard")
def dashboard():
    connection = get_db_connection()
    total_predictions = connection.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()["total"]
    high_count = connection.execute("SELECT COUNT(*) AS count FROM predictions WHERE predicted_performance = 'High'").fetchone()["count"]
    medium_count = connection.execute("SELECT COUNT(*) AS count FROM predictions WHERE predicted_performance = 'Medium'").fetchone()["count"]
    low_count = connection.execute("SELECT COUNT(*) AS count FROM predictions WHERE predicted_performance = 'Low'").fetchone()["count"]
    history = connection.execute(
        """
        SELECT id, student_name, attendance, internal_marks, assignment_score,
               previous_mark, study_hours, completed_assignments,
               predicted_performance, confidence, prediction_time
        FROM predictions
        ORDER BY prediction_time DESC
        LIMIT 20
        """
    ).fetchall()
    connection.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        history=history,
    )


@app.errorhandler(404)
def page_not_found(error):
    return render_template("index.html", errors=["Page not found. Please return to the home page."])


@app.errorhandler(500)
def internal_error(error):
    return render_template("index.html", errors=["An unexpected error occurred. Please try again."])


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
