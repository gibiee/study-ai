# 세계 행복지수 예측

## 프로젝트 소개
- 2015~2017년 세계 행복지수 데이터를 사용하여 2018년 세계 행복지수를 예측한다.
- 다중선형회귀 방식을 사용한다.

## Data
- 2015~2017년 : https://www.kaggle.com/unsdsn/world-happiness
- 2018년 : https://www.kaggle.com/njlow1202/world-happiness-report-data-2018

## 파일 설명

- 1_OLSproject(모든 변수 포함).py
  - statsmodels 패키지의 OLS() 사용.
  - 종속변수를 제외한 모든 변수를 독립변수로 사용.


- 2_OLSproject(Residual 변수 제외).py
  - statsmodels 패키지의 OLS() 사용.
  - 독립변수에서 'Dystopia Residual' 변수를 제외


- 3_Sklearn_project.py
  - scikit-learn 패키지의 LinearRegression() 사용.
  - 독립변수에서 'Dystopia Residual' 변수를 제외


- 4_Tensorflow_project.py
  - tensorflow 라이브러리를 사용하여 경사하강법을 직접 구현.
  - 독립변수에서 'Dystopia Residual' 변수를 제외

## 예측 결과

- 자체 기준 사용 (자세한 기준은 파일 내용 참고)

파일|점수 기준|등수 기준
---|:---:|:---:
1_OLSproject.py|100%|100%
2_OLSproject.py|65.8%|61.9%
3_Sklearn_project.py|65.8%|61.9%
4_Tensorflow_project.py|65.8%|61.9%


## 시사점
- statsmodels의 OLS()를 사용한 예측값과 sklearn의 LinearRegression()을 사용한 예측값이 완벽하게(100%) 일치함. 따라서 두 패키지(statsmodels, sklearn)에서 제공하는 다중선형회귀 함수가 완전히 동일한 알고리즘임을 알 수 있음.
- tensorflow를 사용한 예측값은 위 두 패키지에서의 예측값과 매우 유사함(소수점 둘째 자리까지는 거의 동일). 아마 경사하강법을 더 많이 진행하면 더욱 유사해질 것이라고 생각됨.
- 결국 3가지 패키지(statsmodels, sklearn, tensorflow) 모두 경사하강법 기반의 다중선형회귀 방식이라고 유추해볼 수 있음.

## 참고자료
- https://datascienceschool.net/view-notebook/58269d7f52bd49879965cdc4721da42d/
- https://myjamong.tistory.com/84
