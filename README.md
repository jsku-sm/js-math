# 좌표연구소

특성화고 학생의 기초학력 보완과 탐구 중심 수업을 위한 Streamlit 웹앱입니다.

## 주요 기능

- 두 점을 움직이며 기울기, 직선의 방정식, 거리, 중점 탐구
- 기초·보통·도전 수준의 미션
- 학생 풀이 설명에 대한 규칙 기반 또는 AI 피드백
- 원의 중심과 반지름 변화 탐구
- 학생별 학습 기록 저장
- 교사용 개념별 정답률 및 반복 오답 대시보드
- CSV 다운로드

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 이 폴더의 파일을 GitHub 저장소에 올립니다.
2. Streamlit Community Cloud에서 저장소를 연결합니다.
3. Main file path를 `app.py`로 지정합니다.
4. Deploy를 누릅니다.

## 환경변수

AI 피드백을 사용하려면 다음 환경변수를 설정합니다.

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
TEACHER_PASSWORD=원하는비밀번호
```

API 키가 없으면 규칙 기반 피드백이 자동으로 사용됩니다.

## 유의사항

현재 버전은 학습 기록을 서버의 `learning_data/student_logs.csv`에 저장합니다.
Streamlit Community Cloud에서는 재시작 시 파일이 유지되지 않을 수 있으므로,
실제 장기 운영 시 Google Sheets, Supabase 또는 Firebase 연동을 권장합니다.
