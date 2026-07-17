"""
좌표연구소: 도형의 방정식 탐구 웹앱
실행 방법:
    pip install -r requirements.txt
    streamlit run app.py

선택 기능:
- OpenAI API 키를 환경변수 OPENAI_API_KEY에 등록하면 AI 피드백을 사용할 수 있습니다.
- 키가 없어도 규칙 기반 피드백으로 정상 작동합니다.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


APP_TITLE = "좌표연구소"
DATA_DIR = Path("learning_data")
LOG_FILE = DATA_DIR / "student_logs.csv"
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "1234")

MISSION_BANK = [
    {
        "id": "L1",
        "level": "기초",
        "title": "두 점 사이의 거리",
        "prompt": "점 A(1, 2), B(4, 6) 사이의 거리를 구하세요.",
        "type": "distance",
        "params": {"a": (1, 2), "b": (4, 6)},
        "answer": 5.0,
        "hint": "가로 변화량은 3, 세로 변화량은 4입니다.",
        "concept": "거리 공식",
    },
    {
        "id": "L2",
        "level": "기초",
        "title": "중점 찾기",
        "prompt": "점 A(-2, 4), B(6, 2)의 중점 좌표를 입력하세요.",
        "type": "point",
        "params": {"a": (-2, 4), "b": (6, 2)},
        "answer": (2.0, 3.0),
        "hint": "x좌표끼리, y좌표끼리 각각 더한 뒤 2로 나눕니다.",
        "concept": "중점",
    },
    {
        "id": "L3",
        "level": "기초",
        "title": "기울기 구하기",
        "prompt": "점 A(1, 1), B(5, 3)을 지나는 직선의 기울기를 구하세요.",
        "type": "slope",
        "params": {"a": (1, 1), "b": (5, 3)},
        "answer": 0.5,
        "hint": "y의 변화량을 x의 변화량으로 나눕니다.",
        "concept": "기울기",
    },
    {
        "id": "L4",
        "level": "보통",
        "title": "직선의 방정식",
        "prompt": "기울기가 2이고 점 (1, 3)을 지나는 직선의 y절편을 구하세요.",
        "type": "number",
        "params": {"slope": 2, "point": (1, 3)},
        "answer": 1.0,
        "hint": "y = 2x + b에 점 (1, 3)을 대입하세요.",
        "concept": "직선의 방정식",
    },
    {
        "id": "L5",
        "level": "보통",
        "title": "평행한 직선",
        "prompt": "직선 y = -3x + 2와 평행하고 점 (1, 4)를 지나는 직선의 y절편을 구하세요.",
        "type": "number",
        "params": {"slope": -3, "point": (1, 4)},
        "answer": 7.0,
        "hint": "평행한 두 직선의 기울기는 같습니다.",
        "concept": "평행",
    },
    {
        "id": "L6",
        "level": "보통",
        "title": "원의 방정식",
        "prompt": "중심이 (2, -1)이고 반지름이 3인 원에서 r²의 값을 구하세요.",
        "type": "number",
        "params": {"center": (2, -1), "radius": 3},
        "answer": 9.0,
        "hint": "원의 방정식 오른쪽에는 반지름의 제곱이 들어갑니다.",
        "concept": "원의 방정식",
    },
    {
        "id": "L7",
        "level": "도전",
        "title": "수직인 직선",
        "prompt": "기울기가 2/3인 직선과 수직인 직선의 기울기를 구하세요.",
        "type": "slope",
        "params": {"slope": 2 / 3},
        "answer": -1.5,
        "hint": "수직인 두 직선의 기울기의 곱은 -1입니다.",
        "concept": "수직",
    },
]


def setup_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main-title {font-size:2.1rem; font-weight:800; margin-bottom:.2rem;}
        .sub-title {color:#555; margin-bottom:1.2rem;}
        .concept-box {
            padding: 1rem; border-radius: .8rem; background:#f4f7fb;
            border:1px solid #dfe7f1; margin:.5rem 0;
        }
        .success-box {
            padding: 1rem; border-radius: .8rem; background:#edf9f0;
            border:1px solid #bde5c7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "student_name": "",
        "student_id": "",
        "score": 0,
        "completed_missions": [],
        "attempts": {},
        "reflection_history": [],
        "current_mission_id": "L1",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_log_file() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "student_id",
                    "student_name",
                    "activity",
                    "concept",
                    "result",
                    "attempt",
                    "response",
                    "detail",
                ]
            )


def log_activity(
    activity: str,
    concept: str,
    result: str,
    attempt: int = 0,
    response: str = "",
    detail: str = "",
) -> None:
    ensure_log_file()
    with LOG_FILE.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                st.session_state.get("student_id", ""),
                st.session_state.get("student_name", ""),
                activity,
                concept,
                result,
                attempt,
                response,
                detail,
            ]
        )


def student_ready() -> bool:
    return bool(
        st.session_state.get("student_id", "").strip()
        and st.session_state.get("student_name", "").strip()
    )


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def line_information(x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
    if abs(x2 - x1) < 1e-10:
        return {
            "vertical": True,
            "equation": f"x = {format_number(x1)}",
            "slope": None,
            "intercept": None,
        }

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    sign = "+" if intercept >= 0 else "-"
    equation = (
        f"y = {format_number(slope)}x {sign} {format_number(abs(intercept))}"
    )
    return {
        "vertical": False,
        "equation": equation,
        "slope": slope,
        "intercept": intercept,
    }


def create_coordinate_figure(
    points: list[tuple[float, float, str]],
    lines: list[dict[str, Any]] | None = None,
    circles: list[dict[str, Any]] | None = None,
    x_range: tuple[int, int] = (-10, 10),
    y_range: tuple[int, int] = (-10, 10),
) -> go.Figure:
    fig = go.Figure()

    for x, y, label in points:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[f"{label}({format_number(x)}, {format_number(y)})"],
                textposition="top center",
                marker={"size": 13},
                name=label,
            )
        )

    for line in lines or []:
        if line.get("vertical"):
            x_value = line["x"]
            fig.add_trace(
                go.Scatter(
                    x=[x_value, x_value],
                    y=[y_range[0], y_range[1]],
                    mode="lines",
                    name=line.get("name", "직선"),
                )
            )
        else:
            xs = [x_range[0], x_range[1]]
            ys = [line["slope"] * x + line["intercept"] for x in xs]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=line.get("name", "직선"),
                )
            )

    for circle in circles or []:
        center_x, center_y = circle["center"]
        radius = circle["radius"]
        theta = [i * 2 * math.pi / 180 for i in range(181)]
        xs = [center_x + radius * math.cos(t) for t in theta]
        ys = [center_y + radius * math.sin(t) for t in theta]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=circle.get("name", "원"),
            )
        )

    fig.update_layout(
        height=570,
        xaxis={
            "range": x_range,
            "zeroline": True,
            "gridcolor": "#e8e8e8",
            "scaleanchor": "y",
            "scaleratio": 1,
            "title": "x",
        },
        yaxis={
            "range": y_range,
            "zeroline": True,
            "gridcolor": "#e8e8e8",
            "title": "y",
        },
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="closest",
    )
    return fig


def sidebar_student_info() -> None:
    st.sidebar.header("👤 학습자 정보")
    student_id = st.sidebar.text_input(
        "학번",
        value=st.session_state.student_id,
        placeholder="예: 10101",
    )
    student_name = st.sidebar.text_input(
        "이름",
        value=st.session_state.student_name,
        placeholder="예: 홍길동",
    )

    if st.sidebar.button("학습 시작", use_container_width=True):
        st.session_state.student_id = student_id.strip()
        st.session_state.student_name = student_name.strip()
        if student_ready():
            log_activity("로그인", "학습 시작", "성공")
            st.sidebar.success("학습 기록이 시작되었습니다.")
        else:
            st.sidebar.warning("학번과 이름을 모두 입력하세요.")

    st.sidebar.divider()
    st.sidebar.metric("현재 점수", f"{st.session_state.score}점")
    st.sidebar.metric(
        "완료 미션",
        f"{len(st.session_state.completed_missions)} / {len(MISSION_BANK)}",
    )
    st.sidebar.caption("교사용 기본 비밀번호는 1234입니다. 배포 시 환경변수로 변경하세요.")


def render_explorer() -> None:
    st.header("1. 좌표 탐험")
    st.write(
        "두 점을 직접 움직이며 기울기와 직선의 방정식이 어떻게 달라지는지 관찰하세요."
    )

    control_col, graph_col = st.columns([1, 2])

    with control_col:
        st.subheader("점의 위치")
        x1 = st.slider("A의 x좌표", -8, 8, -3)
        y1 = st.slider("A의 y좌표", -8, 8, -1)
        x2 = st.slider("B의 x좌표", -8, 8, 4)
        y2 = st.slider("B의 y좌표", -8, 8, 5)

        info = line_information(x1, y1, x2, y2)
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx**2 + dy**2)
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

        st.markdown('<div class="concept-box">', unsafe_allow_html=True)
        st.markdown(f"**x의 변화량:** {format_number(dx)}")
        st.markdown(f"**y의 변화량:** {format_number(dy)}")
        if info["vertical"]:
            st.markdown("**기울기:** 정의되지 않음")
        else:
            st.markdown(f"**기울기:** {format_number(info['slope'])}")
        st.markdown(f"**직선의 방정식:** {info['equation']}")
        st.markdown(f"**두 점 사이의 거리:** {format_number(distance)}")
        st.markdown(
            f"**중점:** ({format_number(midpoint[0])}, {format_number(midpoint[1])})"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        observation = st.text_area(
            "관찰한 규칙을 한 문장으로 써 보세요.",
            placeholder="예: 오른쪽으로 갈수록 아래로 내려가면 기울기는 음수이다.",
            key="explorer_observation",
        )
        if st.button("관찰 기록 저장"):
            if not student_ready():
                st.warning("먼저 왼쪽에서 학번과 이름을 입력하세요.")
            elif len(observation.strip()) < 10:
                st.warning("생각을 조금 더 구체적으로 작성하세요.")
            else:
                log_activity(
                    "좌표 탐험",
                    "기울기와 직선",
                    "작성",
                    response=observation,
                    detail=json.dumps(
                        {"A": [x1, y1], "B": [x2, y2], "equation": info["equation"]},
                        ensure_ascii=False,
                    ),
                )
                st.success("관찰 기록을 저장했습니다.")

    with graph_col:
        line = (
            {"vertical": True, "x": x1, "name": info["equation"]}
            if info["vertical"]
            else {
                "vertical": False,
                "slope": info["slope"],
                "intercept": info["intercept"],
                "name": info["equation"],
            }
        )
        fig = create_coordinate_figure(
            points=[
                (x1, y1, "A"),
                (x2, y2, "B"),
                (midpoint[0], midpoint[1], "M"),
            ],
            lines=[line],
        )
        st.plotly_chart(fig, use_container_width=True)

        if info["vertical"]:
            st.info("두 점의 x좌표가 같으므로 직선은 y축과 평행하고 기울기는 정의되지 않습니다.")
        elif info["slope"] > 0:
            st.info("오른쪽으로 갈수록 위로 올라가므로 기울기가 양수입니다.")
        elif info["slope"] < 0:
            st.info("오른쪽으로 갈수록 아래로 내려가므로 기울기가 음수입니다.")
        else:
            st.info("y좌표가 변하지 않으므로 기울기가 0인 수평선입니다.")


def render_missions() -> None:
    st.header("2. 미션 도전")
    st.write("문제를 해결한 뒤, 정답뿐 아니라 풀이 이유도 남겨 보세요.")

    levels = ["전체", "기초", "보통", "도전"]
    selected_level = st.selectbox("난이도", levels)
    available = (
        MISSION_BANK
        if selected_level == "전체"
        else [m for m in MISSION_BANK if m["level"] == selected_level]
    )

    mission_titles = {
        mission["id"]: f"[{mission['level']}] {mission['title']}" for mission in available
    }
    selected_id = st.selectbox(
        "미션 선택",
        options=list(mission_titles.keys()),
        format_func=lambda mission_id: mission_titles[mission_id],
    )
    mission = next(m for m in MISSION_BANK if m["id"] == selected_id)

    st.subheader(mission["title"])
    st.info(mission["prompt"])

    graph = mission_graph(mission)
    if graph:
        st.plotly_chart(graph, use_container_width=True)

    answer: Any
    if mission["type"] == "point":
        col1, col2 = st.columns(2)
        with col1:
            answer_x = st.number_input("x좌표", value=0.0, step=1.0, key=f"x_{selected_id}")
        with col2:
            answer_y = st.number_input("y좌표", value=0.0, step=1.0, key=f"y_{selected_id}")
        answer = (answer_x, answer_y)
    else:
        answer = st.number_input(
            "답 입력",
            value=0.0,
            step=0.5,
            key=f"answer_{selected_id}",
        )

    explanation = st.text_area(
        "어떻게 풀었는지 설명하세요.",
        placeholder="사용한 공식과 계산 과정을 자신의 말로 작성하세요.",
        key=f"explanation_{selected_id}",
    )

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        check_clicked = st.button("정답 확인", use_container_width=True)
    with button_col2:
        hint_clicked = st.button("힌트 보기", use_container_width=True)

    if hint_clicked:
        st.warning(mission["hint"])
        if student_ready():
            log_activity("미션", mission["concept"], "힌트 사용", response=mission["id"])

    if check_clicked:
        if not student_ready():
            st.warning("먼저 왼쪽에서 학번과 이름을 입력하세요.")
            return

        attempts = st.session_state.attempts
        attempts[selected_id] = attempts.get(selected_id, 0) + 1
        attempt_count = attempts[selected_id]
        correct = check_answer(answer, mission["answer"])

        if correct:
            first_completion = selected_id not in st.session_state.completed_missions
            if first_completion:
                points = max(5, 15 - (attempt_count - 1) * 3)
                st.session_state.score += points
                st.session_state.completed_missions.append(selected_id)
            else:
                points = 0

            st.markdown(
                f'<div class="success-box"><b>정답입니다.</b> '
                f'{"+" + str(points) + "점" if points else "이미 완료한 미션입니다."}</div>',
                unsafe_allow_html=True,
            )
            if len(explanation.strip()) < 10:
                st.warning("정답은 맞았지만 풀이 설명이 짧습니다. 사용한 공식을 문장으로 설명해 보세요.")
                explanation_result = "설명 부족"
            else:
                feedback = get_feedback(mission, explanation)
                st.write("#### 설명 피드백")
                st.write(feedback)
                explanation_result = "설명 작성"

            log_activity(
                "미션",
                mission["concept"],
                "정답",
                attempt_count,
                response=str(answer),
                detail=json.dumps(
                    {
                        "mission_id": selected_id,
                        "explanation": explanation,
                        "explanation_result": explanation_result,
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            st.error("아직 정답이 아닙니다. 변화량, 부호, 공식의 순서를 다시 확인하세요.")
            st.caption(f"도움말: {targeted_hint(mission, answer)}")
            log_activity(
                "미션",
                mission["concept"],
                "오답",
                attempt_count,
                response=str(answer),
                detail=json.dumps(
                    {"mission_id": selected_id, "explanation": explanation},
                    ensure_ascii=False,
                ),
            )


def mission_graph(mission: dict[str, Any]) -> go.Figure | None:
    params = mission["params"]
    if "a" in params and "b" in params:
        a = params["a"]
        b = params["b"]
        return create_coordinate_figure(
            points=[(a[0], a[1], "A"), (b[0], b[1], "B")],
            x_range=(-7, 9),
            y_range=(-5, 9),
        )
    if mission["concept"] in {"직선의 방정식", "평행"}:
        slope = params["slope"]
        px, py = params["point"]
        intercept = py - slope * px
        return create_coordinate_figure(
            points=[(px, py, "P")],
            lines=[
                {
                    "vertical": False,
                    "slope": slope,
                    "intercept": intercept,
                    "name": "찾을 직선",
                }
            ],
        )
    if mission["concept"] == "원의 방정식":
        center = params["center"]
        radius = params["radius"]
        return create_coordinate_figure(
            points=[(center[0], center[1], "C")],
            circles=[{"center": center, "radius": radius, "name": "원"}],
        )
    return None


def check_answer(user_answer: Any, correct_answer: Any, tolerance: float = 1e-2) -> bool:
    if isinstance(correct_answer, tuple):
        return all(
            abs(float(user) - float(correct)) <= tolerance
            for user, correct in zip(user_answer, correct_answer)
        )
    return abs(float(user_answer) - float(correct_answer)) <= tolerance


def targeted_hint(mission: dict[str, Any], answer: Any) -> str:
    concept = mission["concept"]
    if concept == "거리 공식":
        return "가로 길이와 세로 길이를 각각 제곱해 더한 뒤 제곱근을 구하세요."
    if concept == "중점":
        return "두 x좌표의 평균과 두 y좌표의 평균을 따로 구하세요."
    if concept == "기울기":
        return "분자는 y의 변화량, 분모는 x의 변화량입니다."
    if concept == "수직":
        return "부호를 바꾸고 분자와 분모를 뒤집어 보세요."
    if concept == "원의 방정식":
        return "반지름 3을 그대로 쓰지 말고 제곱하세요."
    return "주어진 점을 직선의 식 y = mx + b에 대입해 보세요."


def rule_based_feedback(mission: dict[str, Any], explanation: str) -> str:
    text = explanation.strip()
    concept = mission["concept"]
    keywords = {
        "거리 공식": ["거리", "제곱", "제곱근", "변화량"],
        "중점": ["평균", "더", "2", "나누"],
        "기울기": ["변화량", "y", "x", "나누"],
        "직선의 방정식": ["대입", "기울기", "절편", "y"],
        "평행": ["평행", "기울기", "같"],
        "원의 방정식": ["중심", "반지름", "제곱"],
        "수직": ["수직", "곱", "-1", "역수"],
    }
    expected = keywords.get(concept, [])
    found = [word for word in expected if word in text]
    missing = [word for word in expected if word not in text]

    feedback_parts = []
    if len(text) >= 30:
        feedback_parts.append("풀이 과정을 비교적 구체적으로 설명했습니다.")
    else:
        feedback_parts.append("계산 결과뿐 아니라 왜 그 공식을 사용했는지 한 문장 더 써 보세요.")

    if found:
        feedback_parts.append(f"좋은 점: {', '.join(found[:3])}와 관련된 핵심 표현이 들어 있습니다.")
    if missing:
        feedback_parts.append(f"보완할 점: '{missing[0]}'의 의미가 드러나도록 설명해 보세요.")

    feedback_parts.append(
        "마지막에는 ‘따라서 답은 …이다’처럼 결론을 분명히 쓰면 설명이 더 완전해집니다."
    )
    return " ".join(feedback_parts)


def get_feedback(mission: dict[str, Any], explanation: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return rule_based_feedback(mission, explanation)

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
당신은 기초학력이 약한 고등학생을 돕는 친절한 수학교사입니다.
학생의 풀이 설명을 평가하되 정답을 대신 풀어 주지 마세요.

개념: {mission['concept']}
문제: {mission['prompt']}
학생 설명: {explanation}

다음 형식으로 5문장 이내로 답하세요.
1. 잘한 점
2. 빠진 개념 또는 논리
3. 학생이 다시 설명할 수 있도록 주는 한 가지 질문
쉬운 한국어를 사용하세요.
"""
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
        )
        return response.output_text.strip()
    except Exception:
        return rule_based_feedback(mission, explanation)


def render_circle_lab() -> None:
    st.header("3. 원의 방정식 실험실")
    st.write("중심과 반지름을 바꾸면서 원의 방정식과 그래프의 관계를 관찰하세요.")

    controls, graph = st.columns([1, 2])
    with controls:
        h = st.slider("중심의 x좌표 h", -6, 6, 2)
        k = st.slider("중심의 y좌표 k", -6, 6, -1)
        r = st.slider("반지름 r", 1, 6, 3)

        h_sign = "-" if h >= 0 else "+"
        k_sign = "-" if k >= 0 else "+"
        equation = (
            f"(x {h_sign} {abs(h)})² + (y {k_sign} {abs(k)})² = {r**2}"
        )
        st.markdown(f"### {equation}")
        st.write(f"중심: ({h}, {k})")
        st.write(f"반지름: {r}")
        st.caption("식의 괄호 안 부호는 중심 좌표의 부호와 반대입니다.")

        prediction = st.radio(
            "반지름을 크게 하면?",
            ["원이 커진다.", "원이 작아진다.", "중심이 이동한다."],
            index=None,
        )
        if prediction:
            if prediction == "원이 커진다.":
                st.success("맞습니다. 중심은 그대로이고 원의 크기만 커집니다.")
            else:
                st.warning("그래프에서 반지름을 다시 바꾸어 확인해 보세요.")

    with graph:
        fig = create_coordinate_figure(
            points=[(h, k, "C")],
            circles=[{"center": (h, k), "radius": r, "name": equation}],
        )
        st.plotly_chart(fig, use_container_width=True)


def render_reflection() -> None:
    st.header("4. 생각 설명과 성찰")
    st.write("오늘 탐구한 내용을 자신의 말로 설명하면 이해가 더 오래 남습니다.")

    concept = st.selectbox(
        "설명할 개념",
        ["기울기", "직선의 방정식", "두 점 사이의 거리", "중점", "원의 방정식"],
    )
    prompts = {
        "기울기": "기울기의 부호가 그래프의 방향과 어떤 관계가 있는지 설명하세요.",
        "직선의 방정식": "기울기와 y절편이 바뀌면 직선이 어떻게 달라지는지 설명하세요.",
        "두 점 사이의 거리": "거리 공식이 피타고라스 정리와 어떤 관계인지 설명하세요.",
        "중점": "중점 좌표를 평균으로 구하는 이유를 설명하세요.",
        "원의 방정식": "원의 중심과 반지름을 식에서 어떻게 찾는지 설명하세요.",
    }
    st.info(prompts[concept])

    reflection = st.text_area(
        "나의 설명",
        height=160,
        placeholder="‘왜냐하면’, ‘따라서’를 사용해 문장으로 작성해 보세요.",
    )

    if st.button("피드백 받고 저장", use_container_width=True):
        if not student_ready():
            st.warning("먼저 왼쪽에서 학번과 이름을 입력하세요.")
        elif len(reflection.strip()) < 20:
            st.warning("두 문장 이상으로 조금 더 자세히 설명하세요.")
        else:
            dummy_mission = {
                "concept": concept,
                "prompt": prompts[concept],
            }
            feedback = get_feedback(dummy_mission, reflection)
            st.write("#### 피드백")
            st.write(feedback)
            st.session_state.reflection_history.append(
                {
                    "time": datetime.now().strftime("%H:%M"),
                    "concept": concept,
                    "reflection": reflection,
                    "feedback": feedback,
                }
            )
            log_activity(
                "성찰",
                concept,
                "작성",
                response=reflection,
                detail=feedback,
            )
            st.success("설명과 피드백을 저장했습니다.")

    if st.session_state.reflection_history:
        with st.expander("이번 접속에서 작성한 성찰 보기"):
            for item in reversed(st.session_state.reflection_history):
                st.markdown(f"**{item['time']} · {item['concept']}**")
                st.write(item["reflection"])
                st.caption(item["feedback"])
                st.divider()


def render_teacher_dashboard() -> None:
    st.header("5. 교사용 대시보드")
    password = st.text_input("교사용 비밀번호", type="password")
    if password != TEACHER_PASSWORD:
        st.info("비밀번호를 입력하면 학생별 학습 현황을 볼 수 있습니다.")
        return

    ensure_log_file()
    try:
        data = pd.read_csv(LOG_FILE)
    except pd.errors.EmptyDataError:
        st.info("아직 저장된 학습 기록이 없습니다.")
        return

    if data.empty:
        st.info("아직 저장된 학습 기록이 없습니다.")
        return

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    student_names = sorted(data["student_name"].dropna().astype(str).unique())
    selected_students = st.multiselect(
        "학생 선택",
        student_names,
        default=student_names,
    )
    filtered = data[data["student_name"].astype(str).isin(selected_students)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("참여 학생", filtered["student_id"].nunique())
    col2.metric("전체 활동", len(filtered))
    col3.metric("정답 기록", int((filtered["result"] == "정답").sum()))
    col4.metric("오답 기록", int((filtered["result"] == "오답").sum()))

    st.subheader("개념별 정답률")
    mission_logs = filtered[
        (filtered["activity"] == "미션")
        & (filtered["result"].isin(["정답", "오답"]))
    ]
    if not mission_logs.empty:
        summary = (
            mission_logs.assign(correct=(mission_logs["result"] == "정답").astype(int))
            .groupby("concept")
            .agg(시도=("result", "count"), 정답=("correct", "sum"))
            .reset_index()
        )
        summary["정답률(%)"] = (summary["정답"] / summary["시도"] * 100).round(1)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("concept")["정답률(%)"])
    else:
        st.info("미션 풀이 기록이 아직 없습니다.")

    st.subheader("지원이 필요한 학생")
    wrong_logs = filtered[filtered["result"] == "오답"]
    if not wrong_logs.empty:
        support = (
            wrong_logs.groupby(["student_id", "student_name", "concept"])
            .size()
            .reset_index(name="오답 횟수")
            .sort_values("오답 횟수", ascending=False)
        )
        st.dataframe(support.head(20), use_container_width=True, hide_index=True)
    else:
        st.success("현재까지 반복 오답 기록이 없습니다.")

    st.subheader("학생 설명 및 관찰 기록")
    text_logs = filtered[
        filtered["activity"].isin(["성찰", "좌표 탐험"])
    ][["timestamp", "student_id", "student_name", "concept", "response", "detail"]]
    st.dataframe(
        text_logs.sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "학습 기록 CSV 다운로드",
        data=csv_data,
        file_name=f"좌표연구소_학습기록_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_home() -> None:
    st.markdown('<div class="main-title">📐 좌표연구소</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">움직이고, 발견하고, 설명하며 배우는 도형의 방정식</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        이 웹앱은 정답만 맞히는 연습장이 아닙니다.  
        **탐구 → 예측 → 설명 → 피드백 → 다시 탐구**의 순서로 학습합니다.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1. 움직이기**\n\n점과 원을 직접 바꾸며 그래프의 변화를 관찰합니다.")
    with col2:
        st.info("**2. 설명하기**\n\n공식만 쓰지 않고 자신의 말로 이유를 설명합니다.")
    with col3:
        st.info("**3. 다시 도전하기**\n\n힌트와 피드백을 바탕으로 풀이를 수정합니다.")

    st.subheader("오늘의 탐구 질문")
    question = random.choice(
        [
            "두 점의 y좌표가 같으면 기울기는 어떻게 될까?",
            "직선의 기울기가 음수이면 그래프는 어느 방향으로 내려갈까?",
            "원의 중심을 오른쪽으로 옮기면 식의 어느 부분이 달라질까?",
            "두 점의 중점은 왜 좌표의 평균으로 구할 수 있을까?",
        ]
    )
    st.success(question)

    completed = len(st.session_state.completed_missions)
    st.progress(completed / len(MISSION_BANK))
    st.caption(f"미션 진행률: {completed}/{len(MISSION_BANK)}")


def main() -> None:
    setup_page()
    initialize_state()
    ensure_log_file()
    sidebar_student_info()

    menu = st.sidebar.radio(
        "학습 메뉴",
        [
            "홈",
            "좌표 탐험",
            "미션 도전",
            "원의 방정식",
            "생각 설명",
            "교사용 대시보드",
        ],
    )

    if menu == "홈":
        render_home()
    elif menu == "좌표 탐험":
        render_explorer()
    elif menu == "미션 도전":
        render_missions()
    elif menu == "원의 방정식":
        render_circle_lab()
    elif menu == "생각 설명":
        render_reflection()
    else:
        render_teacher_dashboard()


if __name__ == "__main__":
    main()
