from flask import Flask, request, jsonify
import joblib
import numpy as np

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 모델 로드
model = joblib.load('model.pkl')

# 예측을 처리하는 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트로부터 데이터를 받아서 처리
    data = request.json['data']
    data = np.array(data).reshape(1, -1)

    # 모델로 예측 수행
    prediction = model.predict(data)

    # 예측 결과를 JSON으로 반환
    return jsonify({'prediction': prediction.tolist()})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
