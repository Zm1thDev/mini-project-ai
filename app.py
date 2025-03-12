from flask import Flask, render_template, request
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

data = pd.read_csv('water_potability.csv')
data.fillna(data.mean(), inplace=True)

X = data.drop('Potability', axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # รับค่าจากฟอร์ม
            ph = float(request.form['ph'])
            Hardness = float(request.form['Hardness'])
            Solids = float(request.form['Solids'])
            Chloramines = float(request.form['Chloramines'])
            Sulfate = float(request.form['Sulfate'])
            Conductivity = float(request.form['Conductivity'])
            Organic_carbon = float(request.form['Organic_carbon'])
            Trihalomethanes = float(request.form['Trihalomethanes'])
            Turbidity = float(request.form['Turbidity'])

            input_data = pd.DataFrame([[ph, Hardness, Solids, Chloramines, Sulfate, 
                                        Conductivity, Organic_carbon, Trihalomethanes, Turbidity]],
                                      columns=X.columns)

            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)
            predicted_label = "ปลอดภัยในการดื่มน้ำ" if prediction[0] == 1 else "ไม่ปลอดภัยในการดื่มน้ำ"

            return render_template('index.html', result=predicted_label)

        except ValueError:
            return render_template('index.html', error="ข้อมูลไม่ถูกต้อง กรุณาใส่ตัวเลขที่ถูกต้อง")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

