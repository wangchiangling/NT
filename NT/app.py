from flask import Flask, request, render_template
import numpy as np
from catboost import CatBoostRegressor
import pandas as pd

app = Flask(__name__)

# Load the CatBoost model
model_path = 'model/catboost_model.bin'
model = CatBoostRegressor()
model.load_model(model_path)

df_place = pd.read_csv('data/tn_visit_area_info_방문지정보_D.csv')
df_travel = pd.read_csv('data/tn_travel_여행_D.csv')
df_traveler = pd.read_csv('data/tn_traveller_master_여행객 Master_D.csv')

df = pd.merge(df_place, df_travel, on='TRAVEL_ID', how='left')
df = pd.merge(df, df_traveler, on='TRAVELER_ID', how='left')

df_filter = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()  # travle_mission_check가 있는 값들만 가져오자

df_filter.loc[:, 'TRAVEL_MISSION_INT'] = df_filter['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

df_filter = df_filter[[
    'GENDER',
    'AGE_GRP', # 나이
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8', # 스타일
    'TRAVEL_MOTIVE_1', # 목적
    'TRAVEL_COMPANIONS_NUM', # 동반자 수
    'TRAVEL_MISSION_INT', # 여행 미션
    'VISIT_AREA_NM', # 방문지역 이름
    'DGSTFN' # 만족도
]]

# df_filter.loc[:, 'GENDER'] = df_filter['GENDER'].map({'남': 0, '여': 1})

df_filter = df_filter.dropna()

categorical_features_names = [
    'GENDER',
    # 'AGE_GRP',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1',
    # 'TRAVEL_COMPANIONS_NUM',
    'TRAVEL_MISSION_INT',
    'VISIT_AREA_NM',
    # 'DGSTFN',
]

df_filter[categorical_features_names[1:-1]] = df_filter[categorical_features_names[1:-1]].astype(int)
area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()

@app.route('/', methods=['GET', 'POST'])
def index():
    print('dd')
    if request.method == 'POST':
        # Collect form data
        traveler = {
            'GENDER': request.form['GENDER'],
            'AGE_GRP': float(request.form['AGE_GRP']),
            'TRAVEL_STYL_1': int(request.form['TRAVEL_STYL_1']),
            'TRAVEL_STYL_2': int(request.form['TRAVEL_STYL_2']),
            'TRAVEL_STYL_3': int(request.form['TRAVEL_STYL_3']),
            'TRAVEL_STYL_4': int(request.form['TRAVEL_STYL_4']),
            'TRAVEL_STYL_5': int(request.form['TRAVEL_STYL_5']),
            'TRAVEL_STYL_6': int(request.form['TRAVEL_STYL_6']),
            'TRAVEL_STYL_7': int(request.form['TRAVEL_STYL_7']),
            'TRAVEL_STYL_8': int(request.form['TRAVEL_STYL_8']),
            'TRAVEL_MOTIVE_1': int(request.form['TRAVEL_MOTIVE_1']),
            'TRAVEL_COMPANIONS_NUM': float(request.form['TRAVEL_COMPANIONS_NUM']),
            'TRAVEL_MISSION_INT': int(request.form['TRAVEL_MISSION_INT']),
        }

        # Convert to DataFrame for prediction

        results_data = []

        for area in area_names['VISIT_AREA_NM']:
            input_data = list(traveler.values())
            input_data.append(area)

            score = model.predict(input_data)

            results_data.append({'AREA': area, 'SCORE': score})

        results = pd.DataFrame(results_data)
        top_5_recommendations = results.sort_values('SCORE', ascending=False)[:5]        
        print(top_5_recommendations)
        return render_template('index.html', recommendations=top_5_recommendations)

    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)