import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats

# 페이지 기본 설정
st.set_page_config(
    page_title="청소년 아침식사 결식률과 신체활동 실천율 분석",
    page_icon="📊",
    layout="wide"
)

# 파일명 설정 (main.py와 같은 디렉토리에 위치)
PA_FILE = "신체활동_영역_지표_결과_20261116125020.csv"
BF_FILE = "주5일_이상_아침식사_결식률_20261116125001.csv"

@st.cache_data
def load_and_process_data():
    # 인코딩 자동 판별 로드
    try:
        pa_raw = pd.read_csv(PA_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        pa_raw = pd.read_csv(PA_FILE, encoding="cp949")

    try:
        bf_raw = pd.read_csv(BF_FILE, encoding="cp949")
    except UnicodeDecodeError:
        bf_raw = pd.read_csv(BF_FILE, encoding="utf-8")

    years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    genders = {
        "전체": ".1",
        "남학생": ".3",
        "여학생": ".5"
    }
    pa_indicators = {
        2: "하루 60분 주5일 이상 신체활동 실천율",
        3: "주3일 이상 고강도 신체활동 실천율",
        4: "주3일 이상 근력강화운동 실천율"
    }

    # 아침식사 결식률 '소계' 행만 추출
    bf_sub = bf_raw[bf_raw["응답자특성별(2)"] == "소계"].set_index("성별(1)")

    records = []
    for gender, pa_suffix in genders.items():
        for year in years:
            bf_rate = float(bf_sub.loc[gender, f"{year}.1"])
            row = {
                "연도": year,
                "성별": gender,
                "아침식사 결식률(%)": bf_rate
            }
            for row_idx, ind_name in pa_indicators.items():
                row[ind_name] = float(pa_raw.loc[row_idx, f"{year}{pa_suffix}"])
            records.append(row)

    return pd.DataFrame(records), list(pa_indicators.values())

# 데이터 로드
try:
    df, pa_options = load_and_process_data()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다. `main.py`와 같은 폴더에 CSV 파일이 있는지 확인해주세요.\n\n오류 상세: {e}")
    st.stop()

# --- 사이드바: 발표 제어 패널 ---
st.sidebar.header("🎛️ 발표 분석 옵션")
selected_gender = st.sidebar.radio(
    "분석 대상(성별) 선택",
    options=["전체", "남학생", "여학생"],
    index=0
)
selected_pa = st.sidebar.selectbox(
    "신체활동 지표 선택",
    options=pa_options,
    index=0
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**발표 팁:**\n"
    "- '전체' 추세를 먼저 보여준 뒤, '여학생'으로 전환하면 상관계수($r=-0.96$)가 극대화되는 지점을 강조할 수 있습니다."
)

# 선택된 조건으로 데이터 필터링
filtered_df = df[df["성별"] == selected_gender].copy()
x_vals = filtered_df["아침식사 결식률(%)"]
y_vals = filtered_df[selected_pa]

# 피어슨 상관분석 및 선형 회귀 계산
r_val, p_val = stats.pearsonr(x_vals, y_vals)
slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
filtered_df["회귀선"] = slope * x_vals + intercept

# --- 메인 화면 헤더 ---
st.title("📊 청소년 아침식사 결식률과 신체활동 실천율의 상관관계")
st.markdown(f"**탐구 질문:** *청소년의 아침식사 결식률과 신체활동 실천율 사이에는 정말 유의미한 관계가 있는가?* (분석 기준: **2019~2025년 / {selected_gender}**)")
st.markdown("---")

# --- 1. 핵심 통계 요약 카드 ---
col1, col2, col3, col4 = st.columns(4)

bf_diff = x_vals.iloc[-1] - x_vals.iloc[0]
pa_diff = y_vals.iloc[-1] - y_vals.iloc[0]

col1.metric(
    label="2025년 아침식사 결식률",
    value=f"{x_vals.iloc[-1]:.1f}%",
    delta=f"{bf_diff:+.1f}%p (2019년 대비)",
    delta_color="inverse"  # 결식률 증가는 부정적이므로 빨간색 표시
)
col2.metric(
    label=f"2025년 신체활동 실천율",
    value=f"{y_vals.iloc[-1]:.1f}%",
    delta=f"{pa_diff:+.1f}%p (2019년 대비)",
    delta_color="normal"   # 실천율 감소는 빨간색 표시
)
col3.metric(
    label="피어슨 상관계수 (r)",
    value=f"{r_val:.4f}",
    delta="강한 음(-)의 상관관계" if r_val <= -0.7 else "보통 음(-)의 상관관계",
    delta_color="off"
)
col4.metric(
    label="통계적 유의확률 (p-value)",
    value=f"{p_val:.4f}",
    delta="통계적으로 유의미 (p < 0.05)" if p_val < 0.05 else "유의수준 미달 (p ≥ 0.05)",
    delta_color="normal" if p_val < 0.05 else "off"
)

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. 시각화 영역 (2단 구성) ---
left_chart, right_chart = st.columns(2)

# 그래프 1: 연도별 이중축 추세선 (시계열 변화)
with left_chart:
    st.subheader("1️⃣ 연도별 변화 추이 (2019~2025)")
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

    fig_trend.add_trace(
        go.Scatter(
            x=filtered_df["연도"],
            y=filtered_df["아침식사 결식률(%)"],
            name="아침식사 결식률 (%)",
            mode="lines+markers+text",
            text=filtered_df["아침식사 결식률(%)"].apply(lambda v: f"{v:.1f}%"),
            textposition="top center",
            line=dict(color="#EF553B", width=3),
            marker=dict(size=9)
        ),
        secondary_y=False
    )

    fig_trend.add_trace(
        go.Scatter(
            x=filtered_df["연도"],
            y=filtered_df[selected_pa],
            name=f"{selected_pa} (%)",
            mode="lines+markers+text",
            text=filtered_df[selected_pa].apply(lambda v: f"{v:.1f}%"),
            textposition="bottom center",
            line=dict(color="#1F77B4", width=3, dash="solid"),
            marker=dict(size=9, symbol="square")
        ),
        secondary_y=True
    )

    fig_trend.update_layout(
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig_trend.update_yaxes(title_text="아침식사 결식률 (%)", secondary_y=False, range=[25, 52])
    fig_trend.update_yaxes(title_text="신체활동 실천율 (%)", secondary_y=True, range=[0, 75])

    st.plotly_chart(fig_trend, use_container_width=True)

# 그래프 2: 상관관계 산점도 및 회귀선
with right_chart:
    st.subheader("2️⃣ 결식률 vs 신체활동 상관관계 분석")
    fig_scatter = go.Figure()

    # 연도별 실제 데이터 포인트
    fig_scatter.add_trace(
        go.Scatter(
            x=filtered_df["아침식사 결식률(%)"],
            y=filtered_df[selected_pa],
            mode="markers+text",
            text=filtered_df["연도"],
            textposition="top right",
            name="연도별 관측값",
            marker=dict(size=12, color="#636EFA", line=dict(width=1.5, color="white"))
        )
    )

    # 선형 회귀 추세선
    fig_scatter.add_trace(
        go.Scatter(
            x=filtered_df["아침식사 결식률(%)"],
            y=filtered_df["회귀선"],
            mode="lines",
            name=f"추세선 (r = {r_val:.2f})",
            line=dict(color="#00CC96", width=2.5, dash="dash")
        )
    )

    fig_scatter.update_layout(
        height=430,
        xaxis_title="주5일 이상 아침식사 결식률 (%)",
        yaxis_title=f"{selected_pa} (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# --- 3. 발표 결론 및 데이터 해석 가이드 ---
st.markdown("---")
st.subheader("💡 발표 핵심 요약 및 해석 시 유의점")

tab1, tab2, tab3 = st.tabs(["📌 분석 결과 요약", "⚠️ 통계 해석의 한계(중요)", "📋 원본 통합 데이터"])

with tab1:
    st.markdown(f"""
    - **뚜렷한 반비례 관계 확인:** {selected_gender} 기준 2019년부터 2025년까지 아침식사 결식률은 **{x_vals.iloc[0]:.1f}% → {x_vals.iloc[-1]:.1f}%**로 상승한 반면, `{selected_pa}`은 **{y_vals.iloc[0]:.1f}% → {y_vals.iloc[-1]:.1f}%**로 하락했습니다.
    - **통계적 유의성:** 두 변수의 피어슨 상관계수는 **$r = {r_val:.4f}$ ($p = {p_val:.4f}$)**로, 유의수준 $0.05$ 기준에서 {'**통계적으로 유의미한 강한 음의 상관관계**를 보입니다.' if p_val < 0.05 else '유의수준을 충족하지 못했습니다.'}
    - **성별 비교 포인트:** 여학생의 경우 결식률과 신체활동 실천율 간의 음의 상관계수가 $-0.93 \sim -0.96$ 수준으로 남학생($-0.51 \sim -0.83$)보다 훨씬 일관되고 강하게 나타납니다.
    """)

with tab2:
    st.warning("""
    **발표 Q&A 대비용 핵심 포인트 (생태학적 오류 주의):**
    1. **상관관계 ≠ 인과관계:** 두 지표가 반대로 움직인다고 해서 '아침을 굶어서 운동을 안 한다'고 단정할 수는 없습니다. 학업 스트레스 증가, 수면 부족, 스마트폰 사용 시간 증가 같은 **제3의 공통 요인(교란 변수)**이 두 현상을 동시에 유발했을 가능성이 큽니다.
    2. **연도별 집계 데이터(n=7)의 한계:** 이 분석은 개인별 설문 원시데이터(Microdata)가 아닌 7개 연도(2019~2025)의 평균 백분율을 비교한 거시적 추세 분석이므로 표본 수($n=7$)가 작다는 점을 명시하면 발표의 논리적 완성도가 높아집니다.
    """)

with tab3:
    st.dataframe(
        filtered_df[["연도", "성별", "아침식사 결식률(%)", selected_pa]],
        use_container_width=True,
        hide_index=True
    )
