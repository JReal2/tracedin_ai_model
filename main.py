import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import uuid
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import time
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False

# 트랜잭션 데이터 샘플을 생성하는 함수
def generate_sample_data(num_samples=10, anomaly_ratio=0.05, max_spans_per_transaction=20):
    data = []

    # 서비스명, 프로젝트 키, 이름 등 랜덤 값 리스트에서 선택
    service_names = ["auth-service", "payment-service", "order-service", "user-service"]
    project_keys = ["proj-123", "proj-456", "proj-789", "proj-000"]
    span_names = ["GET /api/auth", "POST /api/pay", "GET /api/order", "GET /api/user"]

    # HTTP 응답코드와 메시지
    response_codes = ["200", "201", "404", "500", "403"]
    messages = ["OK", "Created", "Not Found", "Internal Server Error", "Forbidden"]

    current_time = int(time.time() * 1000)  # 현재 시간을 기준으로 Epoch 밀리초로 변환

    for i in range(num_samples):
        # 트랜잭션에 포함될 스팬의 수 랜덤 결정 (최소 1개, 최대 max_spans_per_transaction개)
        num_spans = random.randint(20, max_spans_per_transaction)

        # UUID와 기타 랜덤 값 생성
        trace_id = str(uuid.uuid4())

        service_name = random.choice(service_names)
        project_key = random.choice(project_keys)

        spans = []  # 여러 스팬을 담을 리스트

        for _ in range(num_spans):
            # 스팬별 UUID 생성
            span_id = str(uuid.uuid4())
            parent_span_id = str(uuid.uuid4())
            span_name = random.choice(span_names)

            response_code = random.choice(response_codes)
            message = random.choice(messages)

            # 정상적인 응답 시간: 100ms ~ 2000ms 사이
            normal_duration = random.randint(100, 2000)
            # 이상적인 응답 시간: 3000ms 이상 (일부는 매우 큰 값으로 설정)
            anomaly_duration = random.randint(3000, 10000)

            # 임의로 anomaly_ratio의 비율만큼 이상치 데이터 생성
            if random.random() < anomaly_ratio:
                duration = anomaly_duration
            else:
                duration = normal_duration

            # startEpochMillis와 endEpochMillis 생성
            start_time = current_time + random.randint(0, 10000)  # 최근 10초 사이에 시작한 트랜잭션
            end_time = start_time + duration

            # 스팬 데이터를 생성하여 리스트에 추가
            span = {
                "id": span_id,
                "traceId": trace_id,
                "parentSpanId": parent_span_id,
                "name": span_name,
                "serviceName": service_name,
                "projectKey": project_key,
                "kind": "CLIENT",
                "spanType": "HTTP",
                "startEpochMillis": start_time,
                "endEpochMillis": end_time,
                "duration": duration,
                "startDateTime": pd.to_datetime(start_time, unit='ms').isoformat(),
                "data": {
                    "additionalProp1": {},
                    "additionalProp2": {},
                    "additionalProp3": {}
                },
                "capacity": random.randint(0, 1000),
                "totalAddedValues": random.randint(0, 500)
            }

            spans.append(span)  # 스팬 리스트에 추가

        # 트랜잭션 데이터를 JSON 형태로 구성
        transaction = {
            "path": f"/api/{service_name}/action",
            "responseCode": response_code,
            "message": message,
            "result": {
                "spans": spans,  # 여러 개의 스팬이 포함된 리스트
                "children": [
                    str(uuid.uuid4())  # 자식 스팬도 UUID로 생성
                ]
            },
            "timeStamp": pd.to_datetime(end_time, unit='ms').isoformat()
        }

        data.append(transaction)

        # JSON 데이터를 평탄화하여 DataFrame으로 변환
    df = pd.json_normalize(data, record_path=['result', 'spans'], meta=['path', 'responseCode', 'message', 'timeStamp'])
    return df


# OCSVM을 통한 이상치 탐지
# def detect_anomalies(response_times, threshold=200):
#     # Threshold 이상인 값만 One-Class SVM을 통해 이상치 탐지
#     mask = response_times <= threshold
#
#     response_times_log = np.log1p(response_times).reshape(-1, 1)
#
#     scaler = MinMaxScaler()
#     response_times_scaled = scaler.fit_transform(response_times_log)
#
#     # OCSVM 모델 생성
#     ocsvm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.05)  # nu: 이상치 비율
#     ocsvm.fit(response_times_scaled)
#
#     predictions = ocsvm.predict(response_times_scaled)
#
#     high_indices = np.where(mask)[0]
#     predictions[high_indices] = 1
#
#     save_model(ocsvm)
#
#     return predictions

# OCSVM을 통한 이상치 탐지
def detect_anomalies(spans_df):
    transactions_with_anomalies = []
    grouped = spans_df.groupby('traceId')

    for trace_id, group in grouped:
        response_times = group['duration'].values

        threshold = response_times.mean()

        mask = response_times <= threshold

        response_times_reshaped = response_times.reshape(-1, 1)

        # 스케일링 적용
        scaler = StandardScaler()
        response_times_scaled = scaler.fit_transform(response_times_reshaped)

        # OCSVM 모델 생성
        ocsvm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.2)  # nu: 이상치 비율
        ocsvm.fit(response_times_scaled)

        predictions = ocsvm.predict(response_times_scaled)

        high_indices = np.where(mask)[0]
        predictions[high_indices] = 1 # 해당 인덱스에 predictions_high 값을 반영

        silhouette_avg = silhouette_score(response_times_scaled, predictions)
        print(f"Transaction {trace_id} Silhouette Score: {silhouette_avg:.3f}")

        num_anomalies = sum(predictions == -1)

        if num_anomalies >= 3:
            print(f"Alert: Transaction {trace_id} has {num_anomalies} anomalies!")
            transactions_with_anomalies.append(trace_id)

        normal_data = response_times[predictions == 1]
        normal_index = group.index[predictions == 1]

        anomaly_data = response_times[predictions == -1]
        anomaly_index = group.index[predictions == -1]

        plt.scatter(normal_index, normal_data, label='Normal', color='blue', alpha=0.5)

        plt.scatter(anomaly_index, anomaly_data, label='Anomaly', color='red', alpha=0.7)

        plt.title('Anomaly Detection on Response Times')
        plt.xlabel('Span Index')
        plt.ylabel('Response Time (ms)')
        plt.legend(loc='upper left')
        plt.show()

    return transactions_with_anomalies

def save_model(model, filename='ocsvm_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def load_model(filename='ocsvm_model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

# 메인 로직
if __name__ == "__main__":
    # 샘플 데이터 생성
    api_data = generate_sample_data()

    # 이상치 탐지
    anomalous_transactions = detect_anomalies(api_data)

    print(anomalous_transactions)

    # # 결과 출력
    # if anomalous_transactions:
    #     for transaction in anomalous_transactions:
    #         print(transaction)
    # else:
    #     print("No transactions with 3 or more anomalies detected.")

    # 응답 시간 가져오기
    # response_times = api_data['duration'].values

    # 이상치 탐지
    # anomalies = detect_anomalies(response_times)

    # # 결과 출력
    # for i, (time, anomaly) in enumerate(zip(response_times, anomalies)):
    #     print(
    #         f"Trace ID: {api_data.iloc[i]['id']}, duration: {time} ms, Anomaly: {'Yes' if anomaly == -1 else 'No'}")
    #
    # # 시각화
    # plt.figure(figsize=(10, 6))
    #
    # normal_data = response_times[anomalies == 1]
    # normal_index = np.where(anomalies == 1)[0]
    #
    # anomaly_data = response_times[anomalies == -1]
    # anomaly_index = np.where(anomalies == -1)[0]
    #
    # plt.scatter(normal_index, normal_data, label='Normal', color='blue', alpha=0.5)
    #
    # plt.scatter(anomaly_index, anomaly_data, label='Anomaly', color='red', alpha=0.5)
    #
    # plt.title('Anomaly Detection on Response Times')
    # plt.xlabel('Trace ID')
    # plt.ylabel('duration')
    # plt.show()


