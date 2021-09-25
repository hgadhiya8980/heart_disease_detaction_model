from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")

# df1 = df.loc[0:,['Rank', 'Name', 'City', 'State', 'category','Student_Population','Total_Annual_Cost']]
# df1["Rank"] = df1["Rank"].astype("int")

# cat_cols = df1.select_dtypes(["O"]).keys()

# for var in cat_cols:
#     df1[var].fillna(df1[var].mode()[0], inplace=True)

# df2 = df1["Name"]
# df2 = pd.DataFrame({"Name":df2})
# df3 = df1["City"]
# df3 = pd.DataFrame({"City":df3})
# df4 = df1["State"]
# df4 = pd.DataFrame({"State":df4})

# value = df1["Name"].value_counts().keys()
# value7 = df1["Name"].value_counts().keys()
# value1 = df1["City"].value_counts().keys()
# value2 = df1["State"].value_counts().keys()
# value3 = df1["category"].value_counts().keys()

# for num,var in enumerate(value):
#     num+=1
#     df1["Name"].replace(var, num, inplace=True)

# for num, var in enumerate(value1):
#     num+=1
#     df1["City"].replace(var, num, inplace=True)

# for num,var in enumerate(value2):
#     num+=1
#     df1["State"].replace(var, num, inplace=True)

# for num,var in enumerate(value3):
#     num+=1
#     df1["category"].replace(var, num, inplace=True)

X = df.drop(["target"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=23)

sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("heart_disease_detaction_model.pkl")

def heart_disease_detaction(model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
#     for num,var in enumerate(value):
#         if var == Gender:
#             Gender = num
#     for num1,var1 in enumerate(value1):
#         if var1 == Stream:
#             Stream = num1
            
            
    x = np.zeros(len(X.columns))
#     for num,var in enumerate(X.columns):
#         x[num] = var
    x[0] = age
    x[1] = sex
    x[2] = cp
    x[3] = trestbps
    x[4] = chol
    x[5] = fbs
    x[6] = restecg
    x[7] = thalach
    x[8] = exang
    x[9] = oldpeak
    x[10] = slope
    x[11] = ca
    x[12] = thal
    
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

    
app=Flask(__name__)

@app.route("/")
def home():
    # value7 = list(df2["Name"].value_counts().keys())
    # value7.sort()
    # value10 = list(df3["City"].value_counts().keys())
    # value10.sort()
    # value11 = list(df4["State"].value_counts().keys())
    # value11.sort()
    return render_template("index.html")    #value=value7, value01=value10,value02=value11


@app.route("/predict", methods=["POST"])
def predict():
    age = request.form["age"]
    sex = request.form["sex"]
    cp = request.form["cp"]
    trestbps = request.form["trestbps"]
    chol = request.form["chol"]
    fbs = request.form["fbs"]
    restecg = request.form["restecg"]
    thalach = request.form["thalach"]
    exang = request.form["exang"]
    oldpeak = request.form["oldpeak"]
    slope = request.form["slope"]
    ca = request.form["ca"]
    thal = request.form["thal"]
    
    predicated_price = heart_disease_detaction(model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)


    if predicated_price==1:
        return render_template("index.html", prediction_text="patient has heart disease have No any more.")
    else:
        return render_template("index.html", prediction_text="patient will safe. keep enjoy your life.")

if __name__ == "__main__":
    app.run()    
    