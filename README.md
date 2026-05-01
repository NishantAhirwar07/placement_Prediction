# Placement Prediction ML Model

A simple Machine Learning project that predicts a student's placement package based on their CGPA using **Linear Regression**.

## Overview

This project uses historical placement data to analyze the relationship between **CGPA** and **salary package (LPA)**. The model is trained using Scikit-learn and predicts expected package offers for students.

## Features

- Predict placement package using CGPA
- Uses Linear Regression algorithm
- Data visualization with scatter plot
- Regression line plotting
- Beginner-friendly project
- Suitable for academic submissions

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Dataset

File used: `placement.csv`

Expected columns:

- `cgpa`
- `package`

## Project Workflow

1. Import libraries
2. Load dataset
3. Visualize data
4. Split dataset into training and testing sets
5. Train Linear Regression model
6. Predict package from CGPA
7. Plot regression line

## Model Used

**Linear Regression**

Equation:

`y = mx + c`

Where:

- `y` = Predicted package
- `x` = CGPA
- `m` = Slope
- `c` = Intercept

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Run Project

```bash
python placement_prediction.py
```

## Folder Structure

```text
Placement-Prediction/
├── placement_prediction.py
├── placement.csv
└── README.md
```

## Future Improvements

- Add more input features
- Use advanced regression models
- Deploy using Streamlit
- Build GUI interface
- Add metrics dashboard

## Author

Tony Stark

## License

This project is for educational purposes.
