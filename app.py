import flask
from flask import render_template,request
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle

model = pickle.load(open('model/MLS2_xgb.pkl', 'rb'))
app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('index.html'))

@app.route('/prediction')
def prediction():
    return(flask.render_template('prediction.html'))

@app.route('/projects')
def projects():
    return(flask.render_template('projects.html'))

@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return gm(request.args.get('data'))

@app.route('/visualization')
def visualization():
    feature = "satisfaction"
    bar = gm(feature)
    return render_template('visualization.html', plot=bar)

def gm(feature):
    df = pd.read_csv("airline_passenger_satisfaction.csv") # creating a sample dataframe
    df_fix = df[feature]
    labels = df_fix.unique()
    values = df_fix.value_counts()
    fig = go.Figure()
    if (df_fix.dtypes == 'object'):
        data = [
            go.Pie(labels=labels, values=values, textinfo='label+percent',marker_colors=['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)',
                'rgb(36, 55, 57)', 'rgb(6, 4, 4)'])
        ]
        data2 = [
            go.Box(y=values,
            boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8 # relative position of points wrt box
              )
        ]
        data3 = [
            go.Bar(x=labels, y=values, marker_color='crimson',)
        ]
    elif (df_fix.dtypes == 'int64'):
        data = [
            go.Pie(labels=labels, values=values, hole=.3)
        ]
        data2 = [
            go.Scatter(x=labels, y=values,
                    mode='markers',
                    name='markers')
        ]
        data3 = [
            go.Bar(x=labels, y=values)
        ]
    else:
        data = [
            go.Pie(labels=labels, values=values, hole=.3)
        ]
        data2 = [
            go.Scatter(x=labels, y=values,
                    mode='markers',
                    name='markers')
        ]
        data3 = [
            go.Scatter(x=labels, y=values,
                    mode='markers',
                    name='markers', marker_color='aqua')
        ]
    graphJSON = json.dumps([data, data2, data3], cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphJSON= gm(feature)

    return graphJSON

@app.route('/tes')
def tes():
    return(flask.render_template('tes.html'))

@app.route('/correlation')
def correlation():
    return render_template('correlation.html', graphJSON=fg())

def fg():
    df = pd.read_csv("airline_passenger_satisfaction.csv") # creating a sample dataframe
    enc = OrdinalEncoder()
    df_cat = df.select_dtypes(include='object')
    df_cats = df_cat.columns.tolist()
    enc.fit(df[df_cats])
    df[df_cats] = enc.transform(df[df_cats])
    corr_matrix = df.corr()

    corr = df.corrwith(df['satisfaction']).reset_index()
    corr.columns = ['Index','Correlations']
    corr = corr.set_index('Index')
    corr = corr.sort_values(by=['Correlations'], ascending = True)
    data = [
        go.Heatmap(z=corr_matrix.values, 
        x=corr_matrix.columns, 
        y=corr_matrix.columns,
        colorscale='magma')
    ]
    data2 = [
        go.Heatmap(z=corr.values,
        y=corr.index,
        colorscale='magma')
    ]
    data3 = [
        go.Bar(x=corr_matrix.columns, y=corr_matrix.values[-1],
                base=0,
                marker_color='crimson',
                name='expenses')
    ]
    graphJSON = json.dumps([data, data2, data3], cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/summary')
def summary():
    return(flask.render_template('summary.html'))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    answers = ""
    prediction = model.predict(final_features)
    if prediction[0] == 0:
        answers = "not satisfied"
    else:
        answers = "satisfied"

    output = answers

    return flask.render_template('prediction.html', message = 'The Passenger is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)