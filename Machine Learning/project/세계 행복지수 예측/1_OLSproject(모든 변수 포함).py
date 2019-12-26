import os
os.chdir(os.getcwd() + "/dataset")

import pandas as pd
report2015 = pd.read_csv("world-happiness-report-2015.csv")
report2016 = pd.read_csv("world-happiness-report-2016.csv")
report2017 = pd.read_csv("world-happiness-report-2017.csv")
report2018 = pd.read_csv("world-happiness-report-2018.csv")

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 전처리 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#2015 전처리
report2015.drop('Region', axis=1, inplace=True)
report2015.drop('Standard Error', axis=1, inplace=True)
# 2016
report2016.drop('Region', axis=1, inplace=True)
report2016.drop('Lower Confidence Interval', axis=1, inplace=True)
report2016.drop('Upper Confidence Interval', axis=1, inplace=True)
# 2017
report2017.drop('Whisker.high', axis=1, inplace=True)
report2017.drop('Whisker.low', axis=1, inplace=True)
cols = report2017.columns.tolist()
cols[7], cols[8] = cols[8], cols[7]
report2017 = report2017[cols]
# 2018
cols = report2018.columns.tolist()
cols[0], cols[1] = cols[1], cols[0]
cols[7], cols[8] = cols[8], cols[7]
report2018 = report2018[cols]

# 열 이름 통일
report2015.columns = ['Country', 'Happiness Rank', 'Happiness Score', 'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Government Corruption', 'Generosity', 'Dystopia Residual']
report2016.columns = ['Country', 'Happiness Rank', 'Happiness Score', 'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Government Corruption', 'Generosity', 'Dystopia Residual']
report2017.columns = ['Country', 'Happiness Rank', 'Happiness Score', 'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Government Corruption', 'Generosity', 'Dystopia Residual']
report2018.columns = ['Country', 'Happiness Rank', 'Happiness Score', 'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Government Corruption', 'Generosity', 'Dystopia Residual']

report2018.dropna(inplace=True)
report2018 = report2018.reset_index(drop=True)
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 전처리 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 모델링 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# 2015 ~ 2017년 데이터 통합
report2015_2017 = report2015.append([report2016, report2017], ignore_index=True)

import statsmodels.api as sn
# x는 모든 변수
x = report2015_2017.drop(['Country', 'Happiness Rank', 'Happiness Score'], axis=1)
x = sn.add_constant(x)
y = report2015_2017['Happiness Score']
model = sn.OLS(y,x).fit()
model.summary()

# 첫번째 데이터 행에 대한 예측
model.predict(list(x.loc[0]))
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 모델링 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 예측 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
predict_df = pd.DataFrame(columns=['Predict Score'])
x = report2018.drop(['Country', 'Happiness Rank', 'Happiness Score'], axis=1)
x = sn.add_constant(x)
for i in range(len(report2018)) :
    predict_df.loc[i] = model.predict(list(x.loc[i]))[0]

predict_df['Predict Rank'] = predict_df['Predict Score'].rank(ascending=False)

# 2018년 실제 데이터와 예측 데이터 결합
predict2018 = pd.concat([report2018, predict_df], axis=1)
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 예측 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 정확도 확인 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# 점수 기준으로 결과 확인
error = abs(predict2018['Predict Score'] - predict2018['Happiness Score'])
total = len(error)
correct = len(error[error.values < 0.5])
print("Accurency :", correct/total*100, "%")

# 등수 기준으로 결과 확인
total = len(predict2018)
correct = len(predict2018[abs(predict2018['Happiness Rank'] - predict2018['Predict Rank']) < 15])
print("Accurency :", correct/total*100, "%")
