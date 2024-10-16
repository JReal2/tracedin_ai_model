import river
from river import anomaly
from river import metrics
from river import stream
import random


# 랜덤한 데이터 스트림 생성 함수
def generate_random_data_stream(num_points=100, num_anomalies=5):
    data_stream = []

    # 정상 데이터를 특정 범위 안에서 생성
    for _ in range(num_points - num_anomalies):
        point = {
            "feature_1": random.uniform(0, 3),  # 0 ~ 5 사이의 값
            "feature_2": random.uniform(0, 3)  # 0 ~ 5 사이의 값
        }
        data_stream.append(point)

    # 이상치 데이터를 범위 밖에서 생성
    for _ in range(num_anomalies):
        point = {
            "feature_1": random.uniform(10, 15),  # 10 ~ 15 사이의 값
            "feature_2": random.uniform(10, 15)  # 10 ~ 15 사이의 값
        }
        data_stream.append(point)

    # 데이터 섞기 (정상 데이터와 이상치가 섞이도록)
    random.shuffle(data_stream)

    return data_stream

# 데이터 스트림 생성을 위한 예시 데이터 (특성 2개로 구성된 간단한 데이터)
data_stream = generate_random_data_stream()

# 이상치 탐지를 위한 모델 정의
model = anomaly.HalfSpaceTrees(seed=42, n_trees=25, height=15, window_size=50)

# 이상치 여부 판단을 위한 임계값 설정
threshold = 0.7  # 이상치 확률이 0.7 이상일 경우 이상치로 간주

anomalies = []

# 일정 수준 학습 후 이상치 점수 확인
for i, data_point in enumerate(data_stream):
    model = model.learn_one(data_point)  # 먼저 모델을 학습
    if i >= 5:  # 5개 이상 데이터를 학습한 후부터 점수 계산
        score = model.score_one(data_point)  # 각 데이터 포인트에 대해 이상치 점수 계산
        print(f"Data Point {i}: {data_point}, Anomaly Score: {score:.2f}")

        # 점수가 threshold를 넘는 경우 이상치로 간주
        if score > threshold:
            print(f"--> Anomaly detected at index {i}, data: {data_point}")
            anomalies.append(i)

print(anomalies)


#
# # 실험용 데이터
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2,
#                            n_redundant=0, n_repeated=0, n_classes=2,
#                            n_clusters_per_class=1,
#                            weights=[0.995, 0.005],
#                            class_sep=0.5, random_state=100)
#
# ocs = OneClassSVM(kernel='rbf', # 커널 {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
#           nu=0.05, # Regularization Parameter
#           gamma=0.1 # rbf의 감마
#          ).fit(X)
# outlier_labels = ocs.predict(X)
#
# fig = plt.figure(figsize=(8, 8))
# fig.set_facecolor('white')
# ax = fig.add_subplot()
# ax.scatter(X[:, 0], X[:, 1], c=outlier_labels)
# plt.show()