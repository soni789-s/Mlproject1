from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")
    else:
        data=CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )
        pred_df=data.get_data_as_data()
        print(type(pred_df))
        print(pred_df)

        predictpipeline = PredictPipeline()
        result=predictpipeline.Predict_Data(pred_df.iloc[ : , : ].values)
        return render_template("index.html",results=result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)