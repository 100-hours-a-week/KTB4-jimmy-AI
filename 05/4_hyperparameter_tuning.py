"""
과제 4. 가상 데이터셋을 생성한 뒤, GridSearch와 RandomSearch 기법으로
       하이퍼파라미터 튜닝을 진행하세요.

핵심 아이디어
- GridSearchCV : 지정한 하이퍼파라미터 값들의 "모든 조합"을 전부 시도
                 (조합 수가 많아질수록 시간이 기하급수적으로 늘어남)
- RandomizedSearchCV : 지정한 분포(범위)에서 "무작위로 n_iter개만" 샘플링해 시도
                 (조합 수가 많을 때 Grid보다 훨씬 빠르게 비슷한 성능을 찾을 수 있음)

데이터셋
- sklearn.datasets.make_classification 으로 가상 이진분류 데이터셋 생성
  (04주차 과제에서 했던 가상 데이터셋 생성과 동일한 방식)

모델
- RandomForestClassifier (하이퍼파라미터가 여러 개라 Grid vs Random 비교에 적합)

실행 방법
    pip install scikit-learn
    python 4_hyperparameter_tuning.py
"""

import time

import numpy as np
from scipy.stats import randint
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. 가상 데이터셋 생성
# ------------------------------------------------------------
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# 2. GridSearchCV - 모든 조합을 전부 탐색
# ------------------------------------------------------------
# 아래 그리드는 3 x 3 x 2 x 2 = 36 조합 x 5-fold = 180번 학습
grid_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid=grid_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

start = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

# ------------------------------------------------------------
# 3. RandomizedSearchCV - 같은(혹은 더 넓은) 탐색공간에서 일부만 무작위 샘플링
# ------------------------------------------------------------
# randint(a, b) : a 이상 b 미만의 정수를 균등하게 샘플링하는 분포
random_params = {
    "n_estimators": randint(50, 300),
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=random_params,
    n_iter=36,           # GridSearch와 같은 횟수만큼만 시도해 공정하게 비교
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

start = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start

# ------------------------------------------------------------
# 4. 결과 비교
# ------------------------------------------------------------
if __name__ == "__main__":
    print("===== GridSearchCV =====")
    print(f"탐색한 조합 수      : {len(grid_search.cv_results_['params'])}")
    print(f"최적 하이퍼파라미터  : {grid_search.best_params_}")
    print(f"최적 CV 정확도       : {grid_search.best_score_:.4f}")
    print(f"테스트셋 정확도       : {grid_search.best_estimator_.score(X_test, y_test):.4f}")
    print(f"소요 시간            : {grid_time:.2f}초")

    print("\n===== RandomizedSearchCV =====")
    print(f"탐색한 조합 수      : {len(random_search.cv_results_['params'])}")
    print(f"최적 하이퍼파라미터  : {random_search.best_params_}")
    print(f"최적 CV 정확도       : {random_search.best_score_:.4f}")
    print(f"테스트셋 정확도       : {random_search.best_estimator_.score(X_test, y_test):.4f}")
    print(f"소요 시간            : {random_time:.2f}초")

    print("\n===== 정리 =====")
    print(
        "GridSearch는 모든 조합을 보장하지만 조합 수가 늘어나면 시간이 급증한다.\n"
        "RandomSearch는 같은 시도 횟수로도 더 넓은 탐색공간(연속분포)을 다룰 수 있어,\n"
        "조합이 많을 때 비슷하거나 더 좋은 결과를 더 짧은 시간에 찾는 경우가 많다."
    )
