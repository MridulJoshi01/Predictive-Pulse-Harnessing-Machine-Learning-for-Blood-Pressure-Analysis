from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# since this file lives in the `templates` directory we need to
# tell Flask where to find both templates and static assets
template_dir = os.path.dirname(__file__)
static_dir = os.path.abspath(os.path.join(template_dir, os.pardir, "static"))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load trained model
# best_model.pkl is saved at the project root, not inside templates.
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(base_dir, "best_model.pkl")
model = joblib.load(model_path)

def preprocess_input(form):
    rec = {}
    rec['Gender'] = 0 if form.get('Gender') == 'Male' else 1
    rec['History'] = 1 if form.get('History') == 'Yes' else 0
    rec['Patient'] = 1 if form.get('Patient') == 'Yes' else 0
    rec['TakeMedication'] = 1 if form.get('TakeMedication') == 'Yes' else 0
    rec['BreathShortness'] = 1 if form.get('BreathShortness') == 'Yes' else 0
    rec['VisualChanges'] = 1 if form.get('VisualChanges') == 'Yes' else 0
    rec['NoseBleeding'] = 1 if form.get('NoseBleeding') == 'Yes' else 0
    rec['ControlledDiet'] = 1 if form.get('ControlledDiet') == 'Yes' else 0

    rec['Age'] = {'18-34':1,'35-50':2,'51-64':3,'65+':4}[form.get('Age')]
    rec['Severity'] = {'Mild':0,'Moderate':1,'Severe':2}[form.get('Severity')]
    rec['WhenDiagnosed'] = {'<1 Year':1,'1 - 5 Years':2,'>5 Years':3}[form.get('WhenDiagnosed')]
    rec['Systolic'] = {'100+':0,'100 - 110':0,'111 - 120':1,'121 - 130':2,'130+':3}[form.get('Systolic')]
    rec['Diastolic'] = {'70 - 80':0,'81 - 90':1,'91 - 100':2,'100+':3}[form.get('Diastolic')]

    return np.array([list(rec.values())])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    risk = None
    # always have a value to pass to template, even if GET or error
    risk_class = ""
    advice = ""

    if request.method == 'POST':
        try:
            features = preprocess_input(request.form)
            pred = model.predict(features)[0]

            stage_map = {
                0: "Normal",
                1: "Hypertension (Stage 1)",
                2: "Hypertension (Stage 2)",
                3: "Hypertensive Crisis"
            }

            prediction = stage_map[pred]
            # actionable advice for users
            advice_map = {
                0: "Your levels are normal. Maintain a balanced diet and regular exercise.",
                1: "Consider reducing salt intake and monitoring your BP weekly.",
                2: "Consult a healthcare provider to discuss lifestyle changes or medication.",
                3: "EMERGENCY: Please contact a doctor or visit an ER immediately."
            }
            advice = advice_map[pred]

            if pred == 0:
                risk = "Low Risk"
            elif pred == 1:
                risk = "Moderate Risk"
            elif pred == 2:
                risk = "High Risk"
            else:
                risk = "Critical – Immediate Medical Attention Required"

            # map to a CSS class used for color coding in the template
            risk_class = {
                "Low Risk": "low",
                "Moderate Risk": "moderate",
                "High Risk": "high",
                "Critical – Immediate Medical Attention Required": "critical"
            }[risk]

        except Exception as e:
            prediction = f"Error: {e}"
            risk_class = ""

    return render_template("index.html", prediction=prediction, risk=risk, risk_class=risk_class)

if __name__ == "__main__":
    app.run(debug=True)