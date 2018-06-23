from flask import Flask, render_template, request
import json
import pandas as pd
import os
import errno
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn_pandas import CategoricalImputer
import numpy as np
import pickle
import operator
app = Flask(__name__)


@app.route("/")
def initlogin():
    return render_template("login.html")


# @app.route("/homepage/<user>")
# @app.route("/homepage")
# def gohomepage(user=None):
#     return render_template("homepage.html", user=user)
@app.route("/homepage")
def gohomepage():
    return render_template("homepage.html")


@app.route("/features")
def gofeatures():
    return render_template("featuresVS.html")


@app.route("/getfeatures", methods=['POST', 'GET'])
def getfeatures():
    tf_data = pickle.load(open('topfeatures.p', 'rb'))
    sorted_tf_data = sorted(tf_data.items(), key=operator.itemgetter(1))
    return json.dumps(sorted_tf_data)


@app.route("/donorlist")
def godonorlist():
    return render_template("donor_list.html")


@app.route('/doprediction', methods=['POST', 'GET'])
def doprediction():
    info = request.data
    json_data = json.loads(info)
    meldrange = json_data["meldrange"]
    meldrange = float(meldrange)
    donor_data = json_data["donor"]
    dolen = len(donor_data)
    allrecip_data = json_data["allrecip"]
    allrecip_len = len(allrecip_data)
    donor_df = pd.DataFrame(data=donor_data[1:dolen], columns=donor_data[0])
    allrecip_df = pd.DataFrame(data=allrecip_data[1:allrecip_len], columns=allrecip_data[0])

    filename = 'datafile/donorfile.csv'
    filename2 = 'datafile/recipfile.csv'
    silentremove(filename)
    silentremove(filename2)
    donor_df.to_csv(filename, encoding='utf-8')
    allrecip_df.to_csv(filename2, encoding='utf-8')
    # start to impute --------------------------------------

    donor_df = pd.read_csv('datafile/donorfile.csv', index_col=0)
    recipient_df = pd.read_csv('datafile/recipfile.csv', index_col=0)
    id_df = pd.DataFrame(recipient_df[['recipient_id', 'FINAL_MELD_PELD_LAB_SCORE']])
    X_cf_r = recipient_df.select_dtypes(include=['object'])
    X_ncf_r = recipient_df.select_dtypes(exclude=['object'])

    X_cf_d = donor_df.select_dtypes(include=['object'])
    X_ncf_d = donor_df.select_dtypes(exclude=['object'])

    imp_cat = CategoricalImputer()
    X_cf_r = pd.DataFrame(imp_cat.fit_transform(np.array(X_cf_r)), columns=X_cf_r.columns)

    imp_cat = CategoricalImputer()
    X_cf_d = pd.DataFrame(imp_cat.fit_transform(np.array(X_cf_d)), columns=X_cf_d.columns)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_ncf_r)
    X_ncf_r = pd.DataFrame(imp.transform(X_ncf_r), columns=X_ncf_r.columns)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_ncf_d)
    X_ncf_d = pd.DataFrame(imp.transform(X_ncf_d), columns=X_ncf_d.columns)

    recipient_df = pd.merge(X_ncf_r, X_cf_r, left_index=True, right_index=True)
    # donor_df = pd.merge(X_ncf_d, X_cf_d, left_index=True, right_index=True)

    if meldrange != 200:
        id_df = id_df.loc[(id_df['FINAL_MELD_PELD_LAB_SCORE'] < meldrange) & (id_df['FINAL_MELD_PELD_LAB_SCORE'] >= meldrange - 20)]
        recipient_df = recipient_df.loc[(recipient_df['FINAL_MELD_PELD_LAB_SCORE'] < meldrange) & (recipient_df['FINAL_MELD_PELD_LAB_SCORE'] >= meldrange - 20.0)]

    X_cf_r = recipient_df.select_dtypes(include=['object'])

    X_ncf_r = recipient_df.select_dtypes(exclude=['object'])

    min_max_scaler = preprocessing.MinMaxScaler()
    header = X_ncf_d.columns
    X_ncf_d = min_max_scaler.fit_transform(X_ncf_d)
    X_ncf_d = pd.DataFrame(X_ncf_d, columns=header)

    min_max_scaler = preprocessing.MinMaxScaler()
    header = X_ncf_r.columns
    X_ncf_r = min_max_scaler.fit_transform(X_ncf_r)

    X_ncf_r = pd.DataFrame(X_ncf_r, columns=header)
    X_ncf_r.index = X_cf_r.index
    recipient_df = pd.merge(X_ncf_r, X_cf_r, left_index=True, right_index=True)
    print("recipdf", recipient_df)
    donor_df = pd.merge(X_ncf_d, X_cf_d, left_index=True, right_index=True)


    filename = 'datafile/donorfile.csv'
    filename2 = 'datafile/recipfile.csv'
    filename3 = 'datafile/recipidfile.csv'
    silentremove(filename)
    silentremove(filename2)
    silentremove(filename3)
    donor_df.to_csv(filename, encoding='utf-8')

    print("meldrange",meldrange)
    # if meldrange!=200:
    #     id_df = id_df.loc[(id_df['FINAL_MELD_PELD_LAB_SCORE'] < meldrange) & (id_df['FINAL_MELD_PELD_LAB_SCORE'] >= meldrange - 20)]
    #     recipient_df = recipient_df.loc[(recipient_df['FINAL_MELD_PELD_LAB_SCORE']<meldrange) & (recipient_df['FINAL_MELD_PELD_LAB_SCORE']>= meldrange-20.0)]

    id_df = pd.DataFrame(id_df['recipient_id'],columns=['recipient_id'])


    recipient_df.to_csv(filename2, encoding='utf-8')
    id_df.to_csv(filename3, encoding='utf-8')

    import prediction
    match_score = prediction.matching()
    predict_score = prediction.predictscore()
    return json.dumps({'match': match_score, 'predict': predict_score})


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:
            # errno.ENOENT = no such file or directory
            raise
            # re-raise exception if a different error occurred

if __name__ == "__main__":
    app.run(debug=True)
