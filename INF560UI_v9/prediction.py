import warnings
import pandas as pd
import numpy as np
import sklearn
import scipy
from scipy import stats
from scipy.stats import norm
import datetime as dt
from sklearn import linear_model
import operator
import sklearn_pandas
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn_pandas import CategoricalImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier as RFC
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from keras.models import load_model
import pickle
from scipy.stats import chi2_contingency
warnings.filterwarnings('ignore')


def matching():
    donor_df = pd.read_csv('datafile/donorfile.csv',index_col=0)
    recipient_df = pd.read_csv('datafile/recipfile.csv',index_col=0)
    id_df = pd.read_csv('datafile/recipidfile.csv',index_col=0)

    prob_dict_list = [{('A', 'O'): 0.6946166854565953, ('A', 'A'): 0.7038391224862889, ('A', 'B'): 0.736380713838447, ('A', 'AB'): 0.7446808510638298, ('A', 'A1'): 0.7049907578558225, ('A', 'A2'): 0.6963788300835655, ('A', 'A1B'): 0.7094594594594594, ('A', 'A2B'): 0.7254901960784313, ('O', 'O'): 0.7015718908922792, ('O', 'A'): 0.6923658352229781, ('O', 'B'): 0.6990049751243781, ('O', 'AB'): 0.6886446886446886, ('O', 'A1'): 0.6912356766800867, ('O', 'A2'): 0.700218818380744, ('O', 'A1B'): 0.6621004566210046, ('O', 'A2B'): 0.7045454545454546, ('B', 'O'): 0.7192784667418264, ('B', 'A'): 0.7069825436408977, ('B', 'B'): 0.7119856887298748, ('B', 'AB'): 0.7391304347826086, ('B', 'A1'): 0.7288659793814433, ('B', 'A2'): 0.7077922077922078, ('B', 'A1B'): 0.5769230769230769, ('B', 'A2B'): 0.5833333333333334, ('AB', 'O'): 0.7010416666666667, ('AB', 'A'): 0.6810631229235881, ('AB', 'B'): 0.7285714285714285, ('AB', 'AB'): 0.6451612903225806, ('AB', 'A1'): 0.7073863636363636, ('AB', 'A2'): 0.72, ('AB', 'A1B'): 0.7272727272727273, ('AB', 'A2B'): 0.4, ('A1', 'O'): 0.7419354838709677, ('A1', 'A'): 0.8888888888888888, ('A1', 'B'): 0.5, ('A1', 'AB'): 0.5, ('A1', 'A1'): 0.45454545454545453, ('A1', 'A2'): 1.0, ('A1', 'A1B'): 1.0, ('A1', 'A2B'): 1.0, ('A2', 'O'): 0.5714285714285714, ('A2', 'A'): 0.5, ('A2', 'B'): 0.0, ('A2', 'AB'): 0.5, ('A2', 'A1'): 0.5, ('A2', 'A2'): 0.5, ('A2', 'A1B'): 0.5, ('A2', 'A2B'): 0.5, ('A1B', 'O'): 1.0, ('A1B', 'A'): 1.0, ('A1B', 'B'): 1.0, ('A1B', 'AB'): 0.5, ('A1B', 'A1'): 0.5, ('A1B', 'A2'): 1.0, ('A1B', 'A1B'): 0.5, ('A1B', 'A2B'): 0.5, ('A2B', 'O'): 1.0, ('A2B', 'A'): 0.5, ('A2B', 'B'): 0.5, ('A2B', 'AB'): 0.5, ('A2B', 'A1'): 0.5, ('A2B', 'A2'): 0.5, ('A2B', 'A1B'): 0.5, ('A2B', 'A2B'): 0.5}, {('M', 'M'): 0.7010418040353422, ('M', 'F'): 0.7052882324916627, ('F', 'M'): 0.7032692787621662, ('F', 'F'): 0.6989247311827957}, {('N', 'N'): 0.7041790816716327, ('N', 'P'): 0.68212927756654, ('N', 'ND'): 0.8, ('N', 'U'): 0.6, ('N', 'I'): 0.8333333333333334, ('ND', 'N'): 0.7000553403431101, ('ND', 'P'): 0.6588235294117647, ('ND', 'ND'): 0.6666666666666666, ('ND', 'U'): 0.3333333333333333, ('ND', 'I'): 0.5, ('P', 'N'): 0.699021932424422, ('P', 'P'): 0.7337662337662337, ('P', 'ND'): 0.6875, ('P', 'U'): 0.875, ('P', 'I'): 1.0, ('U', 'N'): 0.7029096477794793, ('U', 'P'): 0.6271186440677966, ('U', 'ND'): 0.8333333333333334, ('U', 'U'): 0.75, ('U', 'I'): 0.5}, {('N', 'N'): 0.7023482680063088, ('N', 'ND'): 0.6949152542372882, ('N', 'P'): 0.8648648648648649, ('N', 'U'): 0.717948717948718, ('N', 'C'): 0.5, ('N', 'I'): 1.0, ('U', 'N'): 0.6933701657458563, ('U', 'ND'): 1.0, ('U', 'P'): 0.6666666666666666, ('U', 'U'): 1.0, ('U', 'C'): 0.5, ('U', 'I'): 0.5, ('P', 'N'): 0.7170138888888888, ('P', 'ND'): 1.0, ('P', 'P'): 0.5, ('P', 'U'): 0.75, ('P', 'C'): 0.5, ('P', 'I'): 0.5, ('ND', 'N'): 0.6773255813953488, ('ND', 'ND'): 0.5, ('ND', 'P'): 0.5, ('ND', 'U'): 0.75, ('ND', 'C'): 0.0, ('ND', 'I'): 0.5}]
    final_score_list = []
    weight_cat_norm = [1, 0.0183, 0.3499, 0.6673]

    recipient_cat = np.array(recipient_df[['ABO','GENDER','HBV_CORE','HBV_SUR_ANTIGEN']])
    donor_cat = np.array(donor_df[['ABO_DON','GENDER_DON','HBV_CORE_DON','HBV_SUR_ANTIGEN_DON']])
    recipient_cont = np.array(recipient_df[['BMI_CALC', 'CREAT_TX', 'TBILI_TX', 'HGT_CM_CALC', 'WGT_KG_CALC']])
    donor_cont = np.array(donor_df[['BMI_DON_CALC', 'CREAT_DON', 'TBILI_DON', 'HGT_CM_DON_CALC', 'WGT_KG_DON_CALC']])

    recipient_weights_cont = [0.8926 * 10, 1 * 10, 0.3142 * 10, 0.7883 * 10, 0.3940 * 10]
    donor_weights_cont = [0.4466 * 10, 0.1846 * 10, 0.2730 * 10, 0.8203 * 10, 0.7958 * 10]

    cat_weights_final = []
    cont_weights_final = []
    final_match = []


    for i in range(len(donor_cat)):
        cat_weights = []
        for j in range(len(recipient_cat)):
            weights = 0

            for k in range(len(recipient_cat[j])):
                try:
                    weights += weight_cat_norm[k] * prob_dict_list[k][
                        (recipient_cat[j][k], donor_cat[i][k])]
                except:
                    weights += 0.0  # when nan is encountered
            cat_weights.append(weights)
        cat_weights_final.append(cat_weights)


    for i in range(len(donor_cont)):
        cont_weights = []
        for j in range(len(recipient_cont)):
            dist = np.sum(((recipient_weights_cont * recipient_cont[j]) - (donor_weights_cont * donor_cont[i])) ** 2, axis=-1,keepdims=True) ** 0.5
            cont_weights.append(dist[0])
        cont_weights_final.append(cont_weights)


    for i in range(len(cat_weights_final)):
        match = []
        for j in range(len(cat_weights_final[i])):
            match.append(round(cat_weights_final[i][j] + cont_weights_final[i][j],3))
        final_match.append(match)


    for i in range(len(final_match)):
        score_list = []
        rec_id = []
        rec_id.append(list(np.array(final_match[i]).argsort())[::-1][:5])

        sorted_list = sorted(final_match[i],reverse = True)
        sorted_df = id_df['recipient_id'].iloc[rec_id[0]]
        count = 0

        for index,row in sorted_df.iteritems():
            score_list.append((row,sorted_list[count]))
            count+=1
        final_score_list.append(score_list)

    print("Final_match",final_score_list)
    return final_score_list

def add_missing_dummy_columns(x_test, xtrain_columns ):
    missing_cols = set(xtrain_columns) - set(x_test.columns)
    for c in missing_cols:
        x_test[c] = 0

def fix_columns(x_test, xtrain_columns ):
    add_missing_dummy_columns( x_test, xtrain_columns )
    # make sure we have all the columns we need
    assert( set( xtrain_columns ) - set( x_test.columns ) == set())
    extra_cols = set( x_test.columns ) - set( xtrain_columns )
    x_test = x_test[ xtrain_columns ]
    return x_test


def predictscore():
    x_test = pd.read_csv('datafile/recipfile.csv', index_col=0)
    id_df = pd.read_csv('datafile/recipidfile.csv', index_col=0)
    final_prediction_score = []
    xtrain_columns = ['NUM_PREV_TX', 'DIAB', 'REM_CD', 'DAYSWAIT_CHRON', 'END_STAT', 'END_BMI_CALC', 'FINAL_ALBUMIN',
                      'FINAL_ASCITES', 'FINAL_BILIRUBIN', 'FINAL_ENCEPH', 'FINAL_INR', 'FINAL_MELD_PELD_LAB_SCORE',
                      'FINAL_SERUM_CREAT', 'FINAL_SERUM_SODIUM', 'TX_PROCEDUR_TY', 'FUNC_STAT_TRR', 'MED_COND_TRR',
                      'PRI_PAYMENT_TRR', 'ON_VENT_TRR', 'ARTIFICIAL_LI_TRR', 'OTH_LIFE_SUP_TRR', 'DA1', 'DA2', 'DB1',
                      'DB2', 'DDR1', 'DDR2', 'AGE_DON', 'COD_CAD_DON', 'DEATH_CIRCUM_DON', 'DEATH_MECH_DON',
                      'BLOOD_INF_DON', 'BUN_DON', 'CREAT_DON', 'PULM_INF_DON', 'SGOT_DON', 'SGPT_DON', 'TBILI_DON',
                      'URINE_INF_DON', 'CANCER_SITE_DON', 'HIST_DIABETES_DON', 'HGT_CM_DON_CALC', 'WGT_KG_DON_CALC',
                      'BMI_DON_CALC', 'ECD_DONOR', 'CREAT_TX', 'TBILI_TX', 'INR_TX', 'ALBUMIN_TX', 'ENCEPH_TX',
                      'ASCITES_TX', 'MELD_PELD_LAB_SCORE', 'EXC_EVER', 'LITYP', 'LOS', 'AGE', 'DIAG', 'ABO_MAT',
                      'COLD_ISCH', 'SHARE_TY', 'HGT_CM_CALC', 'WGT_KG_CALC', 'BMI_CALC', 'DONOR_ID',
                      'TRANSFUS_TERM_DON', 'PH_DON', 'HEMATOCRIT_DON', 'TX_YEAR', 'LISTYR', 'GENDER_F', 'GENDER_M',
                      'ABO_A', 'ABO_A1', 'ABO_A1B', 'ABO_A2', 'ABO_A2B', 'ABO_AB', 'ABO_B', 'ABO_O', 'EXC_HCC_HBL',
                      'EXC_HCC_HCC', 'EXC_HCC_non-HCC', 'EXC_CASE_No', 'EXC_CASE_Yes', 'FINAL_DIALYSIS_PRIOR_WEEK_A',
                      'FINAL_DIALYSIS_PRIOR_WEEK_N', 'FINAL_DIALYSIS_PRIOR_WEEK_Y', 'FINAL_MELD_OR_PELD_MELD',
                      'FINAL_MELD_OR_PELD_PELD', 'MALIG_TRR_N', 'MALIG_TRR_U', 'PORTAL_VEIN_TRR_N', 'PORTAL_VEIN_TRR_U',
                      'PORTAL_VEIN_TRR_Y', 'PREV_AB_SURG_TRR_N', 'PREV_AB_SURG_TRR_U', 'PREV_AB_SURG_TRR_Y',
                      'TIPSS_TRR_N', 'TIPSS_TRR_U', 'TIPSS_TRR_Y', 'HBV_CORE_N', 'HBV_CORE_P', 'HBV_SUR_ANTIGEN_N',
                      'HBV_SUR_ANTIGEN_ND', 'HBV_SUR_ANTIGEN_P', 'HBV_SUR_ANTIGEN_U', 'HCV_SEROSTATUS_N',
                      'HCV_SEROSTATUS_P', 'HCV_SEROSTATUS_U', 'EBV_SEROSTATUS_N', 'EBV_SEROSTATUS_ND',
                      'EBV_SEROSTATUS_P', 'EBV_SEROSTATUS_U', 'HIV_SEROSTATUS_N', 'HIV_SEROSTATUS_U', 'CMV_STATUS_N',
                      'CMV_STATUS_ND', 'CMV_STATUS_P', 'CMV_STATUS_U', 'CMV_IGG_N', 'CMV_IGG_ND', 'CMV_IGG_P',
                      'CMV_IGG_U', 'CMV_IGM_N', 'CMV_IGM_ND', 'CMV_IGM_P', 'CMV_IGM_U', 'PREV_TX_N', 'PREV_TX_Y',
                      'DDAVP_DON_N', 'DDAVP_DON_U', 'DDAVP_DON_Y', 'CMV_DON_I', 'CMV_DON_N', 'CMV_DON_ND', 'CMV_DON_P',
                      'CMV_DON_U', 'HEP_C_ANTI_DON_N', 'HEP_C_ANTI_DON_ND', 'HEP_C_ANTI_DON_P', 'HEP_C_ANTI_DON_U',
                      'HBV_CORE_DON_N', 'HBV_CORE_DON_ND', 'HBV_CORE_DON_P', 'HBV_CORE_DON_U', 'HBV_SUR_ANTIGEN_DON_N',
                      'HBV_SUR_ANTIGEN_DON_ND', 'HBV_SUR_ANTIGEN_DON_P', 'HBV_SUR_ANTIGEN_DON_U', 'DON_TY_C',
                      'DON_TY_L', 'GENDER_DON_F', 'GENDER_DON_M', 'NON_HRT_DON_N', 'NON_HRT_DON_Y', 'ANTIHYPE_DON_N',
                      'ANTIHYPE_DON_U', 'PT_DIURETICS_DON_N', 'PT_DIURETICS_DON_U', 'PT_DIURETICS_DON_Y',
                      'PT_STEROIDS_DON_N', 'PT_STEROIDS_DON_U', 'PT_STEROIDS_DON_Y', 'PT_T3_DON_U', 'PT_T4_DON_N',
                      'PT_T4_DON_U', 'PT_T4_DON_Y', 'VASODIL_DON_N', 'VASODIL_DON_U', 'VASODIL_DON_Y', 'VDRL_DON_N',
                      'VDRL_DON_ND', 'VDRL_DON_P', 'VDRL_DON_U', 'CLIN_INFECT_DON_N', 'CLIN_INFECT_DON_Y',
                      'EXTRACRANIAL_CANCER_DON_N', 'EXTRACRANIAL_CANCER_DON_U', 'HIST_CIG_DON_N', 'HIST_CIG_DON_U',
                      'HIST_CIG_DON_Y', 'HIST_COCAINE_DON_N', 'HIST_COCAINE_DON_U', 'HIST_COCAINE_DON_Y',
                      'DIABETES_DON_N', 'DIABETES_DON_Y', 'HIST_HYPERTENS_DON_N', 'HIST_HYPERTENS_DON_U',
                      'HIST_HYPERTENS_DON_Y', 'HIST_OTH_DRUG_DON_N', 'HIST_OTH_DRUG_DON_Y', 'ABO_DON_A', 'ABO_DON_A1',
                      'ABO_DON_A1B', 'ABO_DON_A2', 'ABO_DON_B', 'ABO_DON_O', 'INTRACRANIAL_CANCER_DON_N',
                      'INTRACRANIAL_CANCER_DON_U', 'INTRACRANIAL_CANCER_DON_Y', 'SKIN_CANCER_DON_U',
                      'SKIN_CANCER_DON_Y', 'HIST_CANCER_DON_N', 'HIST_CANCER_DON_Y', 'PT_OTH_DON_N', 'PT_OTH_DON_Y',
                      'HEPARIN_DON_N', 'HEPARIN_DON_U', 'HEPARIN_DON_Y', 'HBV_CORE_DON.1_N', 'HBV_CORE_DON.1_Y',
                      'INSULIN_DON_N', 'INSULIN_DON_U', 'AGE_GROUP_A', 'AGE_GROUP_P', 'MALIG_N', 'MALIG_U', 'MALIG_Y',
                      'RECOV_OUT_US_N', 'RECOV_OUT_US_Y', 'TATTOOS_N', 'TATTOOS_U', 'TATTOOS_Y', 'LI_BIOPSY_N',
                      'LI_BIOPSY_Y', 'PROTEIN_URINE_N', 'PROTEIN_URINE_U', 'PROTEIN_URINE_Y', 'INOTROP_SUPPORT_DON_N',
                      'INOTROP_SUPPORT_DON_U', 'INOTROP_SUPPORT_DON_Y', 'CDC_RISK_HIV_DON_N', 'CDC_RISK_HIV_DON_Y',
                      'HISTORY_MI_DON_U', 'HISTORY_MI_DON_Y', 'CORONARY_ANGIO_DON_N', 'CORONARY_ANGIO_DON_Y']

    x_test_cfn = x_test.select_dtypes(include=['object'])
    x_test_ncf = x_test.select_dtypes(exclude=['object'])
    #
    # # Handle NULL values in categorical features
    # imp_cat = CategoricalImputer()
    # x_test_cf = pd.DataFrame(imp_cat.fit_transform(np.array(x_test_cf)), columns=x_test_cf.columns)
    #
    # # One-hot encoding for categorical features
    # x_test_cfn = pd.get_dummies(x_test_cf)
    #
    # # Handle NULL values in non-categorical features
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # imp.fit(x_test_ncf)
    # x_test_ncf = pd.DataFrame(imp.transform(x_test_ncf), columns=x_test_ncf.columns)
    #
    # # Scale test data
    # min_max_scaler = preprocessing.MinMaxScaler()
    # header = x_test_ncf.columns
    # x_test_minmax = min_max_scaler.fit_transform(x_test_ncf)
    # x_test_ncf = pd.DataFrame(x_test_minmax, columns=header)
    #
    # merge categorical and non-categorical features
    result = pd.merge(x_test_ncf, x_test_cfn, left_index=True, right_index=True)
    # fix columns
    result = fix_columns(result, xtrain_columns)

    # Eliminate less important features
    less_important_features = ['ABO_DON_A2B', 'ABO_DON_AB', 'ANTIHYPE_DON_Y', 'CARDARREST_NEURO_N',
                               'CARDARREST_NEURO_Y', 'CDC_RISK_HIV_DON_U', 'CLIN_INFECT_DON_U', 'DATA_TRANSPLANT_Y',
                               'DATA_WAITLIST_Y', 'DIABETES_DON_U', 'DIAL_TX_N', 'DIAL_TX_Y',
                               'EXTRACRANIAL_CANCER_DON_Y', 'HBV_CORE_DON.1_U', 'HBV_CORE_DON_I', 'HBV_CORE_ND',
                               'HBV_CORE_U', 'HBV_SUR_ANTIGEN_DON_C', 'HBV_SUR_ANTIGEN_DON_I', 'HCV_SEROSTATUS_ND',
                               'HEP_C_ANTI_DON_C', 'HEP_C_ANTI_DON_I', 'HISTORY_MI_DON_N', 'HIST_CANCER_DON_U',
                               'HIST_OTH_DRUG_DON_U', 'HIV_SEROSTATUS_ND', 'HIV_SEROSTATUS_P', 'INSULIN_DON_Y',
                               'LIFE_SUP_TRR_N', 'LIFE_SUP_TRR_Y', 'LIST_MELD_No', 'LT_ONE_WEEK_DON_N', 'MALIG_TRR_Y',
                               'PT_T3_DON_N', 'PT_T3_DON_Y', 'SKIN_CANCER_DON_N', 'TXLIV_S', 'TXLIV_W', 'TX_MELD_No',
                               'VDRL_DON_I']
    cols = [col for col in result.columns if col not in less_important_features]
    x_test = result[cols]

    # Load Random Forest
    rfmodel_pkl = open('rfmodel.pkl', 'rb')
    clf1_rf = pickle.load(rfmodel_pkl)
    y_pred_clf1 = clf1_rf.predict_proba(x_test)
    y_pred_1 = np.array(y_pred_clf1)
    y_pred_1 = np.delete(y_pred_1, np.s_[:1], 1)

    # Load Logistic Regression
    lrmodel_pkl = open('lrmodel.pkl', 'rb')
    clf2_lr = pickle.load(lrmodel_pkl)
    y_pred_clf2 = clf2_lr.predict_proba(x_test)
    y_pred_2 = np.array(y_pred_clf2)
    y_pred_2 = np.delete(y_pred_2, np.s_[:1], 1)

    # Load Deep Learning Model
    clf3_dl = load_model('dlmodel.h5')
    y_pred_3 = clf3_dl.predict(x_test)
    y_pred_3 = np.array(y_pred_3)

    # ensemble
    ens_arr = np.concatenate((y_pred_1, y_pred_2, y_pred_3), axis=1)
    prob_score = np.mean(ens_arr, axis=1)
    count = 0
    x_test = pd.merge(id_df, x_test, left_index=True, right_index=True)
    for i in x_test['recipient_id'].iteritems():
        final_prediction_score.append((i[1],round(prob_score[count],3)))
        count+=1

    print("final pred score",final_prediction_score)
    return final_prediction_score
