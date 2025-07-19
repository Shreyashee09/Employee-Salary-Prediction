This project uses machine learning techniques to predict whether an employee earns more than $50K/year based on various features like age, education, occupation, etc., using the Adult Census Income dataset.
The dataset used is derived from the **UCI Adult Census Income Dataset**.  
It includes the following features:

- Age  
- Workclass  
- Education & Education-Num  
- Marital-Status  
- Occupation  
- Relationship  
- Race  
- Gender  
- Capital-Gain / Loss  
- Hours-per-week  
- Native-country  
- Salary (Target Variable: `<=50K` or `>50K`)

 Technologies Used

- **Python 3.x**
- **Pandas**, **NumPy** for data manipulation
- **Matplotlib**, **Seaborn** for visualization
- **scikit-learn** for ML modeling
- **Jupyter Notebook / VS Code**

---
 Models Used

- Logistic Regression (baseline model)
- [Optionally: Decision Tree, Random Forest, XGBoost if added later]

---
Steps Performed

1. Data Cleaning (handling missing values like '?')
2. Feature Encoding (Label Encoding for categorical features)
3. Feature Scaling (StandardScaler)
4. Train-Test Split
5. Model Training (Logistic Regression)
6. Evaluation (Accuracy, Precision, Recall, F1-score)

---

 Results

- Accuracy: ~[Your Accuracy Here]%
- [You can add a confusion matrix or plot if you wish]

---

 Future Improvements

- Try ensemble models like Random Forest or XGBoost
- Hyperparameter tuning
- Deploy using Flask / Streamlit
- Add user interface for real-time predictions

---

## 📌 How to Run

```bash
#Clone the repository
git clone https://github.com/yourusername/employee-salary-prediction.git

# Navigate to the project directory
cd employee-salary-prediction

# Install dependencies
pip install -r requirements.txt

# Run the script
python IBM.py
