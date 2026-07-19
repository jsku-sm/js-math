# -*- coding: utf-8 -*-
"""
좌표탐험 (Coordinate Explorer)
특성화고 1학년 대상 - 도형의 방정식 탐구형 학습 웹앱

실행 방법:
    pip install streamlit matplotlib numpy
    streamlit run coordinate_explorer.py
"""

import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 한글 폰트 (환경에 없으면 자동으로 기본 폰트 사용, 에러 방지)
# ----------------------------------------------------------------------
for font_name in ["NanumGothic", "Malgun Gothic", "AppleGothic"]:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue

# ========================================================================
# 페이지 설정 & 스타일
# ========================================================================
st.set_page_config(page_title="좌표탐험 | Coordinate Explorer",
                    page_icon="📍", layout="wide")

PRIMARY = "#2563EB"
ACCENT = "#F97316"
POINT_A = "#EF4444"
POINT_B = "#2563EB"
POINT_C = "#16A34A"
GRID_GRAY = "#E5E7EB"

st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 2rem; max-width: 1200px;}}
    h1 {{font-weight: 800; letter-spacing: -0.5px;}}
    h3 {{color: {PRIMARY};}}
    .stage-badge {{
        display:inline-block; background:{PRIMARY}; color:white;
        border-radius:8px; padding:2px 10px; font-size:14px; font-weight:700;
        margin-right:8px;
    }}
    .goal-box {{
        background:#F8FAFC; border-left:4px solid {PRIMARY};
        padding:12px 16px; border-radius:6px; margin-bottom:14px; font-size:15px;
    }}
    .think-box {{
        background:#FFF7ED; border-left:4px solid {ACCENT};
        padding:12px 16px; border-radius:6px; margin-top:10px; font-size:15px;
    }}
    .result-box {{
        background:#EFF6FF; border-radius:10px; padding:16px 20px;
        font-size:17px; line-height:1.9;
    }}
    div[data-testid="stMetricValue"] {{font-size:26px; color:{PRIMARY};}}
</style>
""", unsafe_allow_html=True)

LIM = 10  # 좌표평면 범위 -10 ~ 10


# ========================================================================
# 공통 함수
# ========================================================================
def base_plot(size=6.2):
    """깔끔한 좌표평면 기본 틀"""
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_GRAY, linewidth=1)
    ax.axhline(0, color="#111827", linewidth=1.3)
    ax.axvline(0, color="#111827", linewidth=1.3)
    ax.set_xticks(range(-LIM, LIM + 1, 2))
    ax.set_yticks(range(-LIM, LIM + 1, 2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=9, colors="#6B7280")
    return fig, ax


def draw_point(ax, x, y, label, color):
    ax.scatter([x], [y], s=90, color=color, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(f"{label}({x:g}, {y:g})", (x, y), textcoords="offset points",
                xytext=(10, 10), fontsize=12, fontweight="bold", color=color)


def stage_header(num, title, goal):
    st.markdown(f'<span class="stage-badge">STEP {num}</span> **{title}**',
                unsafe_allow_html=True)
    st.markdown(f'<div class="goal-box">🎯 <b>학습 목표</b><br>{goal}</div>',
                unsafe_allow_html=True)


def think_box(text):
    st.markdown(f'<div class="think-box">💡 <b>생각해보기</b><br>{text}</div>',
                unsafe_allow_html=True)


def point_sliders(label, default, color, key):
    st.markdown(f"**{label}** (색: <span style='color:{color}'>●</span>)",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    x = c1.slider(f"{label} x", -LIM, LIM, default[0], 1, key=f"{key}x")
    y = c2.slider(f"{label} y", -LIM, LIM, default[1], 1, key=f"{key}y")
    return x, y


# ========================================================================
# STEP 1. x축 위, 두 점 사이의 거리
# ========================================================================
def _s1_reset():
    st.session_state.s1_A_input = 0
    st.session_state.s1_B_input = 0
    st.session_state.s1_A_confirmed = None
    st.session_state.s1_B_confirmed = None
    st.session_state.s1_show_result = False


def _s1_apply():
    st.session_state.s1_A_confirmed = st.session_state.s1_A_input
    st.session_state.s1_B_confirmed = st.session_state.s1_B_input
    st.session_state.s1_show_result = False


def stage1():
    stage_header(1, "x축 위, 두 점 사이의 거리",
                 "x축 위의 두 점 A, B의 좌표를 입력하고, 두 점 사이의 거리를 구하는 공식을 확인합니다.")

    st.session_state.setdefault("s1_A_input", 0)
    st.session_state.setdefault("s1_B_input", 0)
    st.session_state.setdefault("s1_A_confirmed", None)
    st.session_state.setdefault("s1_B_confirmed", None)
    st.session_state.setdefault("s1_show_result", False)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("**점 A, B의 x값 입력하기**")
        c1, c2 = st.columns(2)
        c1.number_input("점 A의 x값", -LIM, LIM, step=1, key="s1_A_input")
        c2.number_input("점 B의 x값", -LIM, LIM, step=1, key="s1_B_input")

        cb1, cb2 = st.columns(2)
        cb1.button("✅ 입력값 적용", key="s1_apply_btn", use_container_width=True, on_click=_s1_apply)
        cb2.button("🔄 Reset", key="s1_reset_btn", use_container_width=True, on_click=_s1_reset)

        st.markdown("---")
        ready = st.session_state.s1_A_confirmed is not None and st.session_state.s1_B_confirmed is not None
        if st.button("📐 두 점 사이의 거리 계산", key="s1_calc", disabled=not ready, use_container_width=True):
            st.session_state.s1_show_result = True

        if ready and st.session_state.s1_show_result:
            A, B = st.session_state.s1_A_confirmed, st.session_state.s1_B_confirmed
            dist = abs(B - A)
            st.markdown(f"""
<div class="result-box">
두 점 A({A:g}, 0), B({B:g}, 0)은 모두 x축 위에 있으므로<br>
거리 = |x<sub>B</sub> − x<sub>A</sub>| = |{B:g} − {A:g}| = <b>{dist:g}</b>
</div>
""", unsafe_allow_html=True)
            think_box("두 점이 모두 x축 위에 있을 때는 y좌표가 둘 다 0이라서 계산에서 사라집니다. 만약 두 점이 y축 위에 있다면 거리는 어떻게 구할까요?")
        elif not ready:
            st.info("A, B 두 점의 x값을 입력하고 '입력값 적용' 버튼을 눌러주세요.")

    with col2:
        fig, ax = base_plot()
        if st.session_state.s1_A_confirmed is not None:
            draw_point(ax, st.session_state.s1_A_confirmed, 0, "A", POINT_A)
        if st.session_state.s1_B_confirmed is not None:
            draw_point(ax, st.session_state.s1_B_confirmed, 0, "B", POINT_B)
        if st.session_state.s1_A_confirmed is not None and st.session_state.s1_B_confirmed is not None:
            A, B = st.session_state.s1_A_confirmed, st.session_state.s1_B_confirmed
            ax.plot([A, B], [0, 0], color="#111827", linewidth=2.6, zorder=4)
        st.pyplot(fig)


# ========================================================================
# STEP 2. 두 점 사이의 거리
# ========================================================================
def _s2_reset():
    st.session_state.s2_Ax_input = -4
    st.session_state.s2_Ay_input = -2
    st.session_state.s2_Bx_input = 4
    st.session_state.s2_By_input = 4
    st.session_state.s2_A_confirmed = None
    st.session_state.s2_B_confirmed = None
    st.session_state.s2_show_result = False


def _s2_apply():
    st.session_state.s2_A_confirmed = (st.session_state.s2_Ax_input, st.session_state.s2_Ay_input)
    st.session_state.s2_B_confirmed = (st.session_state.s2_Bx_input, st.session_state.s2_By_input)
    st.session_state.s2_show_result = False


def stage2():
    stage_header(2, "두 점 사이의 거리",
                 "두 점 A, B의 좌표를 입력하면 좌표평면에 점이 표시되고, "
                 "직각삼각형을 통해 두 점 사이의 거리를 구하는 공식을 확인합니다.")

    st.session_state.setdefault("s2_Ax_input", -4)
    st.session_state.setdefault("s2_Ay_input", -2)
    st.session_state.setdefault("s2_Bx_input", 4)
    st.session_state.setdefault("s2_By_input", 4)
    st.session_state.setdefault("s2_A_confirmed", None)
    st.session_state.setdefault("s2_B_confirmed", None)
    st.session_state.setdefault("s2_show_result", False)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown(f"**점 A** (색: <span style='color:{POINT_A}'>●</span>)", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.number_input("A의 x값", -LIM, LIM, step=1, key="s2_Ax_input")
        c2.number_input("A의 y값", -LIM, LIM, step=1, key="s2_Ay_input")

        st.markdown(f"**점 B** (색: <span style='color:{POINT_B}'>●</span>)", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        c3.number_input("B의 x값", -LIM, LIM, step=1, key="s2_Bx_input")
        c4.number_input("B의 y값", -LIM, LIM, step=1, key="s2_By_input")

        cb1, cb2 = st.columns(2)
        cb1.button("✅ 입력값 적용", key="s2_apply_btn", use_container_width=True, on_click=_s2_apply)
        cb2.button("🔄 Reset", key="s2_reset_btn", use_container_width=True, on_click=_s2_reset)

        st.markdown("---")
        ready = st.session_state.s2_A_confirmed is not None and st.session_state.s2_B_confirmed is not None
        if st.button("📐 두 점 사이의 거리 계산", key="s2_calc", disabled=not ready, use_container_width=True):
            st.session_state.s2_show_result = True

        if ready and st.session_state.s2_show_result:
            ax_, ay_ = st.session_state.s2_A_confirmed
            bx_, by_ = st.session_state.s2_B_confirmed
            dx, dy = bx_ - ax_, by_ - ay_
            dist = (dx**2 + dy**2) ** 0.5
            m1, m2, m3 = st.columns(3)
            m1.metric("가로 (dx)", f"{abs(dx):g}")
            m2.metric("세로 (dy)", f"{abs(dy):g}")
            m3.metric("거리", f"{dist:.2f}")
            st.markdown(f"""
<div class="result-box">
AB = √(dx² + dy²) = √(({bx_:g}−{ax_:g})² + ({by_:g}−{ay_:g})²) = √{dx**2 + dy**2:g} ≈ <b>{dist:.2f}</b>
</div>
""", unsafe_allow_html=True)
            think_box("A와 B를 잇는 직각삼각형에서 빗변이 바로 '두 점 사이의 거리'입니다. 피타고라스 정리와 어떤 관계가 있을까요?")
        elif not ready:
            st.info("A, B 두 점의 좌표를 입력하고 '입력값 적용' 버튼을 눌러주세요.")

    with col2:
        fig, ax = base_plot()
        if st.session_state.s2_A_confirmed is not None:
            draw_point(ax, *st.session_state.s2_A_confirmed, "A", POINT_A)
        if st.session_state.s2_B_confirmed is not None:
            draw_point(ax, *st.session_state.s2_B_confirmed, "B", POINT_B)
        if st.session_state.s2_A_confirmed is not None and st.session_state.s2_B_confirmed is not None:
            ax_, ay_ = st.session_state.s2_A_confirmed
            bx_, by_ = st.session_state.s2_B_confirmed
            ax.plot([ax_, bx_], [ay_, by_], color="#111827", linewidth=2.4, zorder=4)
            ax.plot([ax_, bx_], [ay_, ay_], "--", color="#9CA3AF", linewidth=1.6)
            ax.plot([bx_, bx_], [ay_, by_], "--", color="#9CA3AF", linewidth=1.6)
        st.pyplot(fig)



# ========================================================================
# STEP 3. 내분점
# ========================================================================
def stage3():
    stage_header(3, "내분점", "선분 AB 위의 점 P가 어떤 비율로 나누는지 슬라이더로 조작해봅니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        ax_, ay_ = point_sliders("점 A", (-6, -3), POINT_A, "s3a")
        bx_, by_ = point_sliders("점 B", (6, 5), POINT_B, "s3b")
        c1, c2 = st.columns(2)
        m = c1.slider("m (비율)", 1, 9, 2, 1, key="s3m")
        n = c2.slider("n (비율)", 1, 9, 1, 1, key="s3n")
        px = (m * bx_ + n * ax_) / (m + n)
        py = (m * by_ + n * ay_) / (m + n)
        st.markdown("---")
        st.markdown(f"""
<div class="result-box">
P는 AB를 <b>{m} : {n}</b> 으로 내분<br>
P = ( (m·x₂+n·x₁)/(m+n) , (m·y₂+n·y₁)/(m+n) )<br>
P = ( <b>{px:.2f}</b> , <b>{py:.2f}</b> )
</div>
""", unsafe_allow_html=True)
        think_box("m과 n을 같게 만들면 P는 어디로 이동하나요? 그 점의 이름은 무엇일까요? (STEP 5에서 확인!)")

    with col2:
        fig, ax = base_plot()
        draw_point(ax, ax_, ay_, "A", POINT_A)
        draw_point(ax, bx_, by_, "B", POINT_B)
        ax.plot([ax_, bx_], [ay_, by_], color="#D1D5DB", linewidth=2)
        draw_point(ax, round(px, 2), round(py, 2), "P", ACCENT)
        st.pyplot(fig)


# ========================================================================
# STEP 4. 외분점
# ========================================================================
def stage4():
    stage_header(4, "외분점", "선분 AB의 연장선 위에서 점 Q가 나타나는 위치를 관찰합니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        ax_, ay_ = point_sliders("점 A", (-4, -2), POINT_A, "s4a")
        bx_, by_ = point_sliders("점 B", (4, 3), POINT_B, "s4b")
        c1, c2 = st.columns(2)
        m = c1.slider("m (비율)", 1, 9, 3, 1, key="s4m")
        n = c2.slider("n (비율, m과 달라야 함)", 1, 9, 1, 1, key="s4n")
        if m == n:
            st.warning("⚠ m과 n이 같으면 외분점을 만들 수 없어요! (평행선이 되어버려요) 값을 다르게 조정해보세요.")
            qx, qy = None, None
        else:
            qx = (m * bx_ - n * ax_) / (m - n)
            qy = (m * by_ - n * ay_) / (m - n)
            st.markdown("---")
            st.markdown(f"""
<div class="result-box">
Q는 AB를 <b>{m} : {n}</b> 으로 외분<br>
Q = ( (m·x₂-n·x₁)/(m-n) , (m·y₂-n·y₁)/(m-n) )<br>
Q = ( <b>{qx:.2f}</b> , <b>{qy:.2f}</b> )
</div>
""", unsafe_allow_html=True)
        think_box("외분점은 선분 '안'이 아니라 '바깥'에 생깁니다. m>n일 때와 m<n일 때 Q는 어느 쪽 바깥에 생길까요?")

    with col2:
        fig, ax = base_plot()
        # 연장선
        dx, dy = bx_ - ax_, by_ - ay_
        ext = np.array([-3, 3])
        ax.plot([ax_ - dx * 2, bx_ + dx * 2], [ay_ - dy * 2, by_ + dy * 2],
                "--", color="#D1D5DB", linewidth=1.5)
        ax.plot([ax_, bx_], [ay_, by_], color="#111827", linewidth=2)
        draw_point(ax, ax_, ay_, "A", POINT_A)
        draw_point(ax, bx_, by_, "B", POINT_B)
        if m != n and -LIM <= qx <= LIM and -LIM <= qy <= LIM:
            draw_point(ax, round(qx, 2), round(qy, 2), "Q", ACCENT)
        st.pyplot(fig)


# ========================================================================
# STEP 5. 중점
# ========================================================================
def stage5():
    stage_header(5, "중점", "선분 AB를 정확히 절반으로 나누는 점을 찾아봅니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        ax_, ay_ = point_sliders("점 A", (-5, 2), POINT_A, "s5a")
        bx_, by_ = point_sliders("점 B", (5, -4), POINT_B, "s5b")
        mx, my = (ax_ + bx_) / 2, (ay_ + by_) / 2
        st.markdown("---")
        st.markdown(f"""
<div class="result-box">
중점 M = ( (x₁+x₂)/2 , (y₁+y₂)/2 )<br>
M = ( ({ax_:g}+{bx_:g})/2 , ({ay_:g}+{by_:g})/2 ) = ( <b>{mx:g}</b> , <b>{my:g}</b> )
</div>
""", unsafe_allow_html=True)
        think_box("중점은 내분점 공식에서 m=n=1일 때와 같습니다. STEP 3에서 확인했던 내용과 비교해보세요.")

    with col2:
        fig, ax = base_plot()
        draw_point(ax, ax_, ay_, "A", POINT_A)
        draw_point(ax, bx_, by_, "B", POINT_B)
        ax.plot([ax_, bx_], [ay_, by_], color="#D1D5DB", linewidth=2)
        draw_point(ax, mx, my, "M", ACCENT)
        st.pyplot(fig)


# ========================================================================
# STEP 6. 무게중심
# ========================================================================
def stage6():
    stage_header(6, "삼각형의 무게중심", "세 점을 움직여 삼각형을 만들고 무게중심의 위치를 확인합니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        ax_, ay_ = point_sliders("점 A", (-6, -4), POINT_A, "s6a")
        bx_, by_ = point_sliders("점 B", (6, -4), POINT_B, "s6b")
        cx_, cy_ = point_sliders("점 C", (0, 6), POINT_C, "s6c")
        gx, gy = (ax_ + bx_ + cx_) / 3, (ay_ + by_ + cy_) / 3
        st.markdown("---")
        st.markdown(f"""
<div class="result-box">
무게중심 G = ( (x₁+x₂+x₃)/3 , (y₁+y₂+y₃)/3 )<br>
G = ( <b>{gx:.2f}</b> , <b>{gy:.2f}</b> )
</div>
""", unsafe_allow_html=True)
        think_box("삼각형을 아무리 다르게 만들어도 세 중선은 항상 G에서 만납니다. 실제로 중선을 그어 확인해볼까요?")

    with col2:
        fig, ax = base_plot()
        tri = plt.Polygon([(ax_, ay_), (bx_, by_), (cx_, cy_)],
                           closed=True, fill=True, facecolor="#DBEAFE", edgecolor="#111827", linewidth=2)
        ax.add_patch(tri)
        draw_point(ax, ax_, ay_, "A", POINT_A)
        draw_point(ax, bx_, by_, "B", POINT_B)
        draw_point(ax, cx_, cy_, "C", POINT_C)
        # 중선
        mab = ((ax_ + bx_) / 2, (ay_ + by_) / 2)
        ax.plot([cx_, mab[0]], [cy_, mab[1]], "--", color="#9CA3AF", linewidth=1.3)
        draw_point(ax, gx, gy, "G", ACCENT)
        st.pyplot(fig)


# ========================================================================
# STEP 7. 직선의 방정식
# ========================================================================
def stage7():
    stage_header(7, "직선의 방정식", "두 점을 지나는 직선을 만들고, 기울기와 방정식이 실시간으로 바뀌는 것을 관찰합니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        ax_, ay_ = point_sliders("점 A", (-4, -2), POINT_A, "s7a")
        bx_, by_ = point_sliders("점 B", (4, 4), POINT_B, "s7b")
        st.markdown("---")
        if bx_ == ax_:
            st.markdown(f"""
<div class="result-box">
두 점의 x좌표가 같으므로 <b>수직선</b>입니다.<br>
방정식: <b>x = {ax_:g}</b>
</div>
""", unsafe_allow_html=True)
        else:
            slope = (by_ - ay_) / (bx_ - ax_)
            b_val = ay_ - slope * ax_
            st.markdown(f"""
<div class="result-box">
기울기 a = (y₂-y₁)/(x₂-x₁) = ({by_:g}-{ay_:g})/({bx_:g}-{ax_:g}) = <b>{slope:.2f}</b><br>
y절편 b = <b>{b_val:.2f}</b><br>
직선의 방정식: <b>y = {slope:.2f}x + {b_val:.2f}</b>
</div>
""", unsafe_allow_html=True)
        think_box("A를 위로, B를 오른쪽으로 움직여보세요. 기울기 부호(+/-)가 그래프 모양과 어떻게 연결되나요?")

    with col2:
        fig, ax = base_plot()
        if bx_ == ax_:
            ax.axvline(ax_, color=PRIMARY, linewidth=2.4)
        else:
            slope = (by_ - ay_) / (bx_ - ax_)
            b_val = ay_ - slope * ax_
            xs = np.array([-LIM, LIM])
            ys = slope * xs + b_val
            ax.plot(xs, ys, color=PRIMARY, linewidth=2.4)
        draw_point(ax, ax_, ay_, "A", POINT_A)
        draw_point(ax, bx_, by_, "B", POINT_B)
        st.pyplot(fig)


# ========================================================================
# STEP 8. 원의 방정식
# ========================================================================
def stage8():
    stage_header(8, "원의 방정식", "중심과 반지름을 조작하여 원의 방정식이 어떻게 만들어지는지 확인합니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        cx_, cy_ = point_sliders("중심 C", (1, 1), ACCENT, "s8c")
        r = st.slider("반지름 r", 1, 8, 4, 1, key="s8r")
        st.markdown("---")
        sign_x = "-" if cx_ >= 0 else "+"
        sign_y = "-" if cy_ >= 0 else "+"
        st.markdown(f"""
<div class="result-box">
원의 방정식: (x {sign_x} {abs(cx_):g})² + (y {sign_y} {abs(cy_):g})² = {r:g}²<br>
즉, (x {sign_x} {abs(cx_):g})² + (y {sign_y} {abs(cy_):g})² = <b>{r**2:g}</b>
</div>
""", unsafe_allow_html=True)
        think_box("중심이 원점(0,0)일 때 방정식이 어떻게 간단해지나요? 반지름을 2배로 늘리면 우변은 몇 배가 될까요?")

    with col2:
        fig, ax = base_plot()
        theta = np.linspace(0, 2 * np.pi, 200)
        xs = cx_ + r * np.cos(theta)
        ys = cy_ + r * np.sin(theta)
        ax.plot(xs, ys, color=PRIMARY, linewidth=2.4)
        ax.fill(xs, ys, color="#DBEAFE", alpha=0.4)
        ax.plot([cx_, cx_ + r], [cy_, cy_], "--", color=ACCENT, linewidth=1.6)
        draw_point(ax, cx_, cy_, "C", ACCENT)
        st.pyplot(fig)


# ========================================================================
# STEP 9. 원과 직선의 위치관계
# ========================================================================
def stage9():
    stage_header(9, "원과 직선의 위치관계", "직선을 움직여 원과 만나는 점의 개수(0개/1개/2개)를 관찰합니다.")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.markdown(f"**원** (색: <span style='color:{ACCENT}'>●</span>)", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cx_ = c1.slider("중심 x", -LIM, LIM, 0, 1, key="s9cx")
        cy_ = c2.slider("중심 y", -LIM, LIM, 0, 1, key="s9cy")
        r = c3.slider("반지름", 1, 8, 4, 1, key="s9r")

        st.markdown(f"**직선 y = ax + b** (색: <span style='color:{PRIMARY}'>●</span>)", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        a = c4.slider("기울기 a", -5.0, 5.0, 1.0, 0.5, key="s9a")
        b_val = c5.slider("y절편 b", -LIM, LIM, 0, 1, key="s9b")

        # 원의 중심에서 직선까지의 거리: |a*cx - cy + b| / sqrt(a^2+1)
        d = abs(a * cx_ - cy_ + b_val) / (a**2 + 1) ** 0.5
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("중심-직선 거리 (d)", f"{d:.2f}")
        c2.metric("반지름 (r)", f"{r:g}")

        if abs(d - r) < 0.05:
            rel, detail = "접한다 (한 점에서 만남)", "d = r"
        elif d < r:
            rel, detail = "서로 다른 두 점에서 만난다", "d < r"
        else:
            rel, detail = "만나지 않는다", "d > r"

        st.markdown(f"""
<div class="result-box">
판정: <b>d {detail.split()[1]} r</b> → 원과 직선은 <b>{rel}</b>
</div>
""", unsafe_allow_html=True)
        think_box("d와 r이 정확히 같아지는 순간, 그래프에서 직선이 원에 '스치듯' 닿는 모습을 찾아보세요. 이게 바로 접선입니다!")

    with col2:
        fig, ax = base_plot()
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx_ + r * np.cos(theta), cy_ + r * np.sin(theta), color=ACCENT, linewidth=2.2)
        xs = np.array([-LIM, LIM])
        ax.plot(xs, a * xs + b_val, color=PRIMARY, linewidth=2.2)
        draw_point(ax, cx_, cy_, "C", ACCENT)

        # 교점 계산 및 표시
        A_ = 1 + a**2
        B_ = 2 * (a * (b_val - cy_) - cx_)
        C_ = cx_**2 + (b_val - cy_) ** 2 - r**2
        disc = B_**2 - 4 * A_ * C_
        if disc >= 0:
            x1 = (-B_ + disc**0.5) / (2 * A_)
            x2 = (-B_ - disc**0.5) / (2 * A_)
            for i, xi in enumerate([x1, x2] if disc > 0.01 else [x1]):
                yi = a * xi + b_val
                ax.scatter([xi], [yi], s=70, color="#111827", zorder=6)
        st.pyplot(fig)


# ========================================================================
# 메인 레이아웃
# ========================================================================
STAGES = {
    "1. x축에서의 거리": stage1,
    "2. 두 점 사이의 거리": stage2,
    "3. 내분점": stage3,
    "4. 외분점": stage4,
    "5. 중점": stage5,
    "6. 무게중심": stage6,
    "7. 직선의 방정식": stage7,
    "8. 원의 방정식": stage8,
    "9. 원과 직선의 위치관계": stage9,
}

st.title("📍 좌표탐험 (Coordinate Explorer)")
st.caption("슬라이더를 움직여 도형의 방정식을 직접 발견해보세요.")

with st.sidebar:
    st.markdown("### 🗂 학습 단계")
    choice = st.radio("단계를 선택하세요", list(STAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**사용 방법**\n\n① 슬라이더로 점/직선/원을 움직이기\n\n② 오른쪽 그래프 변화 관찰\n\n③ 아래 공식·결과 확인\n\n④ '생각해보기' 질문에 답해보기")

STAGES[choice]()
