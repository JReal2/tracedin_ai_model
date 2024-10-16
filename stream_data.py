from kafka import KafkaConsumer
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_classification
import joblib

# 실험용 데이터
initial_train_data, _ = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.995, 0.005],
                           class_sep=0.5, random_state=100)


# 사전 학습된 OCSVM 모델 불러오기 (예시)
ocsvm = OneClassSVM(kernel='rbf', # 커널 {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
          nu=0.05, # Regularization Parameter
          gamma=0.1 # rbf의 감마
         )
ocsvm.fit(initial_train_data)  # 초기 학습

# KafkaConsumer 설정 (서버와 연결 및 토픽 구독)
consumer = KafkaConsumer('real_time_data_topic', bootstrap_servers=['localhost:9092'])

# 데이터 배치
batch_data = []

for message in consumer:
    data = np.frombuffer(message.value, dtype=np.float32).reshape(1, -1)
    prediction = ocsvm.predict(data)
    batch_data.append(data)

    if prediction == -1:
        print(f"Outlier detected: {data}")
    else:
        print(f"Normal data: {data}")

    # 일정 개수 이상의 데이터가 모이면 배치 학습 진행
    if len(batch_data) > 100:
        batch_data_np = np.vstack(batch_data)
        ocsvm.fit(batch_data_np)  # 주기적으로 모델 업데이트
        batch_data = []  # 배치 초기화

# 모델 저장
joblib.dump(ocsvm, 'ocsvm_model.pkl')