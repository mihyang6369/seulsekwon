import streamlit as st
import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
from geopy.distance import geodesic
import os
import requests
from dotenv import load_dotenv
from streamlit_folium import st_folium
import re

# ==========================================
# 1. 환경 설정 및 상수 정의
# ==========================================

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 웹 페이지의 제목과 아이콘을 설정합니다.
st.set_page_config(page_title="서울 슬세권 지수 대시보드", page_icon="🏙️", layout="wide")

# 디자인 시스템 색상 정의
PRIMARY_COLOR = "#3b82f6"     # 메인 블루
SECONDARY_COLOR = "#1e293b"   # 다크 네이비 (텍스트/타이틀)
ACCENT_COLOR = "#6366f1"      # 포인트 인디고
BACKGROUND_COLOR = "#f8fafc"  # 배경 연회색
CARD_BG = "#ffffff"           # 카드 배경

# 카테고리별 이모지 매핑
EMOJI_MAP = {
    "스타벅스": "☕", "편의점": "🏪", "세탁소": "🏪", "마트": "🏪", "대형마트": "🏬",
    "백화점": "🏬", "버스정류장": "🚌", "지하철역": "🚇", "병원": "🏥", "의원": "💊",
    "약국": "💊", "경찰서": "🚓", "파출소": "🚓", "도서관": "📚", "서점": "📚",
    "학교": "🏫", "공원": "🌳", "체육시설": "🏋️", "은행": "🏦", "금융": "🏦"
}

# 분석용 대분류 카테고리 그룹 설정
CATEGORY_GROUPS = {
    "생활/편의🏪": ["스타벅스", "편의점", "세탁소", "마트", "대형마트", "백화점"],
    "교통🚌": ["버스정류장", "지하철역"],
    "의료💊": ["병원", "의원", "약국"],
    "안전/치안🚨": ["경찰서", "파출소"],
    "교육/문화📚": ["도서관", "서점", "학교"],
    "자연/여가🌳": ["공원", "체육시설"],
    "금융🏦": ["은행", "금융"]
}

# 프리미엄 디자인을 위한 커스텀 스타일(CSS) 설정
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: {BACKGROUND_COLOR};
    }}
    
    /* 대시보드 카드 디자인 (글래스모피즘) */
    .dashboard-card {{
        background: {CARD_BG};
        padding: 1.5rem;
        border-radius: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 1.2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .dashboard-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 28px -5px rgba(0, 0, 0, 0.08);
    }}
    
    /* 지수 강조 스타일 */
    .score-container {{
        text-align: center;
        padding: 1.5rem;
    }}
    
    .metric-value {{
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }}
    
    /* 등급 배지 디자인 */
    .grade-badge-s {{ background-color: #f59e0b; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3); }}
    .grade-badge-a {{ background-color: #10b981; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3); }}
    .grade-badge-b {{ background-color: #3b82f6; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3); }}
    .grade-badge-c {{ background-color: #64748b; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; box-shadow: 0 4px 10px rgba(100, 116, 139, 0.3); }}
    
    /* 버튼 및 입력창 디자인 고도화 */
    div.stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 0.8rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    div.stButton > button:hover {{
        box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4);
        transform: scale(1.02);
        opacity: 0.95;
    }}
    
    .stTextInput > div > div > input {{
        border-radius: 0.8rem;
        border: 1px solid #e2e8f0;
        padding: 0.6rem 1rem;
    }}
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {{
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 위치 정보 관련 함수
# ==========================================

def get_coordinates(query, api_key):
    """카카오 로컬 API를 사용하여 장소명으로 위경도 좌표를 가져옵니다."""
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": query}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if result['documents']:
                address_info = result['documents'][0]
                return {
                    "address_name": address_info['address_name'],
                    "lat": address_info['y'],
                    "lng": address_info['x']
                }
        return None
    except Exception as e:
        st.error(f"좌표 검색 중 오류 발생: {e}")
        return None

def get_coords_from_address(address: str):
    """주소 텍스트를 위도/경도 튜플로 변환합니다."""
    # 1. Streamlit Secrets에서 먼저 찾고, 없으면 환경 변수(os.getenv)에서 찾습니다.
    api_key = None
    if "KAKAO_REST_API_KEY" in st.secrets:
        api_key = st.secrets["KAKAO_REST_API_KEY"]
    else:
        api_key = os.getenv("KAKAO_REST_API_KEY")

    if not api_key:
        st.error("⚠️ 카카오 API 키를 찾을 수 없습니다. Streamlit Cloud의 Secrets 설정이나 .env 파일을 확인해주세요.")
        return None
        
    result = get_coordinates(address, api_key)
    if isinstance(result, dict):
        return float(result['lat']), float(result['lng'])
    return None

def get_dong_name(address):
    """주소에서 행정동 이름을 추출합니다."""
    if not isinstance(address, str): return "알 수 없음"
    match = re.search(r'([가-힣]+동)', address)
    if match: return match.group(1)
    return "서울시 전체"

# ==========================================
# 3. 데이터 로드 및 분석 함수
# ==========================================

@st.cache_data
def load_all_data():
    """클린징된 서울시 기초 데이터를 모두 로드하여 통합 데이터프레임을 생성합니다."""
    # 배포 환경과 로컬 환경 모두 호환되도록 경로를 탐색합니다.
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cleaned"),
        os.path.join("data", "cleaned"),
        "data/cleaned"
    ]
    
    base_path = None
    for p in possible_paths:
        if os.path.exists(p):
            base_path = p
            break
            
    if not base_path:
        st.error("🚨 데이터 폴더(data/cleaned)를 찾을 수 없습니다. GitHub에 데이터가 포함되어 있는지 확인해주세요.")
        return pd.DataFrame(columns=['name', 'lat', 'lon', 'sub_category', 'address'])
    
    file_map = {
        "starbucks_seoul_cleaned.csv": "스타벅스", "bus_station_seoul_cleaned.csv": "버스정류장",
        "metro_station_seoul_cleaned.csv": "지하철역", "hospital_seoul_cleaned.csv": "병원",
        "police_seoul_cleaned_ver2.csv": "경찰서", "library_seoul_cleaned.csv": "도서관",
        "bookstore_seoul_cleaned.csv": "서점", "school_seoul_cleaned.csv": "학교",
        "park_raw_cleaned_revised.csv": "공원", "finance_seoul_cleaned.csv": "은행",
        "large_scale_shop_seoul_cleaned.csv": "대형마트", "sosang_seoul_cleaned.csv": "소상공인"
    }
    
    all_dfs = []
    for file, sub_cat in file_map.items():
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            encodings = ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except: continue
            
            if df is not None:
                if sub_cat == "소상공인": df['sub_category'] = df['카테고리_소']
                else: df['sub_category'] = sub_cat
                name_col = '상호명' if '상호명' in df.columns else ('점포명' if '점포명' in df.columns else '이름')
                if name_col in df.columns and '위도' in df.columns and '경도' in df.columns:
                    temp_df = df[[name_col, '위도', '경도', 'sub_category']].copy()
                    temp_df.columns = ['name', 'lat', 'lon', 'sub_category']
                    if '주소' in df.columns: temp_df['address'] = df['주소']
                    all_dfs.append(temp_df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=['name', 'lat', 'lon', 'sub_category', 'address'])

def calculate_seulsekwon_index(center_lat, center_lon, data, weights, radius_m):
    """중심 좌표와 반경 정보를 바탕으로 슬세권 지수를 산출합니다."""
    radius_km = radius_m / 1000.0
    scores, counts, nearby_facilities = {}, {}, []
    if data.empty or 'lat' not in data.columns:
        return 0.0, {cat: 0.0 for cat in CATEGORY_GROUPS.keys()}, {cat: 0 for cat in CATEGORY_GROUPS.keys()}, []

    # 지수 산출을 위한 배점 기준
    max_counts = {"생활/편의🏪": 15, "교통🚌": 10, "의료💊": 5, "안전/치안🚨": 2, "교육/문화📚": 5, "자연/여가🌳": 5, "금융🏦": 5}
    
    # 박스 필터링 (속도 최적화)
    lat_margin, lon_margin = radius_km / 111.0, radius_km / 88.0
    mask = (data['lat'] >= center_lat - lat_margin) & (data['lat'] <= center_lat + lat_margin) & \
           (data['lon'] >= center_lon - lon_margin) & (data['lon'] <= center_lon + lon_margin)
    filtered_data = data[mask].copy()

    for group_name, sub_cats in CATEGORY_GROUPS.items():
        group_data = filtered_data[filtered_data['sub_category'].apply(lambda x: any(sc in str(x) for sc in sub_cats))]
        actual_count = 0
        for _, row in group_data.iterrows():
            dist = geodesic((center_lat, center_lon), (row['lat'], row['lon'])).meters
            if dist <= radius_m:
                actual_count += 1
                row_dict = row.to_dict(); row_dict['distance'] = dist; row_dict['group'] = group_name
                found_emoji = "📍"
                for key, emoji in EMOJI_MAP.items():
                    if key in str(row['sub_category']): found_emoji = emoji; break
                row_dict['emoji'] = found_emoji
                nearby_facilities.append(row_dict)
        counts[group_name] = actual_count
        m = max_counts.get(group_name, 10)
        score = (min(actual_count, m) / m) * weights.get(group_name, 0)
        scores[group_name] = round(score, 2)

    return round(sum(scores.values()), 1), scores, counts, nearby_facilities

def create_visualizations(total_score, scores, counts, facilities, dong_name):
    """세련된 Plotly 테마를 활용한 시각화 자료를 제작합니다."""
    viz = {}
    
    # 세련된 폰트 및 배경 설정
    layout_opts = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=SECONDARY_COLOR),
        margin=dict(l=30, r=30, t=50, b=30)
    )

    # 1. 고화질 레이더 차트
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(scores.values()) + [list(scores.values())[0]],
        theta=list(scores.keys()) + [list(scores.keys())[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color=ACCENT_COLOR, width=3),
        marker=dict(size=8, color=ACCENT_COLOR),
        name='지수 분포'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 35], gridcolor="#f1f5f9", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="#f1f5f9", tickfont=dict(size=12, weight='bold'))
        ),
        showlegend=False,
        **layout_opts
    )
    viz['radar'] = fig_radar

    # 2. 정밀 게이지 차트
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = total_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': SECONDARY_COLOR},
            'bar': {'color': ACCENT_COLOR},
            'bgcolor': "#f8fafc",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 70], 'color': "#fee2e2"},
                {'range': [70, 85], 'color': "#fef3c7"},
                {'range': [85, 100], 'color': "#dcfce7"}]
        }
    ))
    fig_gauge.update_layout(height=280, **layout_opts)
    viz['gauge'] = fig_gauge

    # 3. 데이터 비교 바 차트 (그라데이션 스타일 미지원으로 색상 최적화)
    fig_compare = px.bar(
        x=[f"'{dong_name}'", "서울 평균"],
        y=[total_score, 75.5],
        color=[f"'{dong_name}'", "서울 평균"],
        color_discrete_map={f"'{dong_name}'": PRIMARY_COLOR, "서울 평균": "#cbd5e1"},
        text_auto='.1f'
    )
    fig_compare.update_traces(marker_line_width=0, opacity=0.9)
    fig_compare.update_layout(showlegend=False, xaxis_title="", yaxis_title="지수", height=320, **layout_opts)
    viz['compare'] = fig_compare

    # 4. 인프라 비중 도넛 차트
    fig_pie = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        hole=.6,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_pie.update_layout(height=320, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.3), **layout_opts)
    viz['pie'] = fig_pie

    # 5. 시설 분포 트리맵
    if facilities:
        f_df = pd.DataFrame(facilities)
        fig_tree = px.treemap(f_df, path=['group', 'sub_category', 'name'], values='distance', color='group',
                              color_discrete_sequence=px.colors.qualitative.Safe)
        fig_tree.update_layout(**layout_opts)
        fig_tree.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        viz['tree'] = fig_tree
    
    return viz

def create_enhanced_map(lat, lon, facilities, radius_m):
    """커스텀 스타일링된 지도를 생성합니다."""
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles="cartodbpositron", control_scale=True)
    folium.Circle([lat, lon], radius=radius_m, color=PRIMARY_COLOR, weight=2, fill=True, fill_opacity=0.08, 
                  tooltip=f"분석 반경 ({radius_m}m)").add_to(m)
    
    # 홈 마커
    folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='home', prefix='fa'), tooltip="분석 중심").add_to(m)

    # 시설물 마커 디테일
    for f in facilities[:100]:
        html = f"""
        <div style="font-size: 16px; background: white; border-radius: 50%; width: 30px; height: 30px; 
                    display: flex; align-items: center; justify-content: center; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 2.5px solid {ACCENT_COLOR};">
            {f['emoji']}
        </div>
        """
        folium.Marker([f['lat'], f['lon']], icon=folium.DivIcon(html=html), popup=f"<b>{f['name']}</b><br>{f['sub_category']}<br>거리: {f['distance']:.0f}m").add_to(m)
    return m

# ==========================================
# 4. Streamlit UI 로직
# ==========================================

# 데이터 로딩 애니메이션
if 'data' not in st.session_state:
    with st.status("🚀 분석 엔진 초기화 중...", expanded=True) as status:
        st.write("📍 서울시 지리 데이터 로드 중...")
        st.session_state.data = load_all_data()
        
        if st.session_state.data.empty:
            st.error("❌ 데이터를 로드하지 못했습니다. 파일 경로와 인코딩을 확인해주세요.")
            status.update(label="초기화 실패", state="error", expanded=True)
        else:
            st.write(f"📊 {len(st.session_state.data):,}개의 데이터 포인트 로드 완료.")
            status.update(label="준비 완료!", state="complete", expanded=False)

# 세션 상태 기본값
state_defaults = {
    'coords': (37.5665, 126.9780), 'address': "서울특별시 중구 세종대로 110", 'radius': 500,
    'weights': {"생활/편의🏪": 30, "교통🚌": 20, "의료💊": 15, "안전/치안🚨": 10, "교육/문화📚": 5, "자연/여가🌳": 15, "금융🏦": 5}
}
for key, val in state_defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# 헤더 섹션
st.markdown(f"""
    <div style="text-align: center; padding: 3rem 0 2rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; color: {SECONDARY_COLOR}; letter-spacing: -1.5px; margin-bottom: 0.5rem;">
            🏙️ <span style="color: {PRIMARY_COLOR};">SEOUL</span> SEULSEKWON
        </h1>
        <p style="color: #64748b; font-size: 1.25rem; font-weight: 400;">살기 좋은 동네의 기준, 주변 5분 거리 인프라를 한눈에.</p>
    </div>
""", unsafe_allow_html=True)

# 검색 섹션 (심플 & 모던)
with st.container():
    col_s1, col_s2, col_s3 = st.columns([2.5, 1, 1])
    with col_s1: query = st.text_input("📍 분석할 주소 또는 장소명", placeholder="강남역, 한남동 더힐, 서울시청 등...")
    with col_s2: rad = st.select_slider("📏 분석 반경 (m)", options=[300, 500, 700, 1000, 1500], value=st.session_state.radius)
    with col_s3: st.write("<div style='height:28px;'></div>", unsafe_allow_html=True); search = st.button("지수 산출하기")

if search and query:
    with st.spinner("🔍 위치를 파악하고 있습니다..."):
        res = get_coords_from_address(query)
        if res:
            st.session_state.coords, st.session_state.address, st.session_state.radius = res, query, rad
            st.rerun()
        else: st.error("❌ 유효한 주소를 찾을 수 없습니다. 다시 시도해 주세요.")

# 메인 분석 결과
if st.session_state.address:
    st.markdown("<hr style='border: 0; height: 1px; background: #e2e8f0; margin: 3rem 0;'>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(f"<h2 style='color: {SECONDARY_COLOR}; font-weight: 700;'>⚖️ 지수 가중치 커스텀</h2>", unsafe_allow_html=True)
        st.info("나에게 중요한 항목의 가중치를 조절해 보세요. 총합 100점 기반으로 계산됩니다.")
        new_w = {}
        for cat, val in st.session_state.weights.items():
            new_w[cat] = st.slider(cat, 0, 50, val)
        st.write("---")
        if st.button("변경 설정으로 재계산"):
            st.session_state.weights = new_w; st.rerun()

    # 계산 엔진 가동
    t_score, scores, counts, facilities = calculate_seulsekwon_index(
        st.session_state.coords[0], st.session_state.coords[1], st.session_state.data, st.session_state.weights, st.session_state.radius
    )
    dong = get_dong_name(st.session_state.address)
    viz = create_visualizations(t_score, scores, counts, facilities, dong)

    # 1층: 종합 점수 및 메인 지표
    col_r1, col_r2, col_r3 = st.columns([1.1, 1, 0.9])
    
    with col_r1:
        grade, badge_cls = "C", "grade-badge-c"
        if t_score >= 90: grade, badge_cls = "S (최상)", "grade-badge-s"
        elif t_score >= 80: grade, badge_cls = "A (우수)", "grade-badge-a"
        elif t_score >= 70: grade, badge_cls = "B (보통)", "grade-badge-b"
        
        st.markdown(f"""
            <div class="dashboard-card score-container">
                <p style="color: #94a3b8; font-weight: 600; font-size: 0.9rem; text-transform: uppercase;">Current Analysis Area</p>
                <h3 style="color: {SECONDARY_COLOR}; margin-top: 0;">{st.session_state.address}</h3>
                <div class="metric-value">{t_score}</div>
                <div style="margin-top: 1.5rem;"><span class="{badge_cls}">{grade} GRADE</span></div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_r2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {SECONDARY_COLOR}; text-align: center; margin-bottom: -15px;'>영역별 지표 분석</h4>", unsafe_allow_html=True)
        st.plotly_chart(viz['radar'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_r3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        # 게이지 차트 제목은 게이지 함수 내부에 있으므로 여기선 생략 또는 커스텀
        st.markdown(f"<h4 style='color: {SECONDARY_COLOR}; text-align: center; margin-bottom: -15px;'>슬세권 도달률</h4>", unsafe_allow_html=True)
        st.plotly_chart(viz['gauge'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2층: 지도 및 통계 데이터
    col_m1, col_m2 = st.columns([1.6, 1])
    
    with col_m1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {SECONDARY_COLOR}; margin-bottom: 1.5rem;'>🗺️ 인터랙티브 시설 분포 Map</h4>", unsafe_allow_html=True)
        m = create_enhanced_map(st.session_state.coords[0], st.session_state.coords[1], facilities, st.session_state.radius)
        map_out = st_folium(m, width="100%", height=550, key="main_map")
        
        # 지도 클릭 인터랙션
        if map_out and map_out.get("last_clicked"):
            nc = (map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"])
            if round(nc[0], 5) != round(st.session_state.coords[0], 5):
                st.session_state.coords = nc; st.session_state.address = f"지도 클릭 위치 ({nc[0]:.4f}, {nc[1]:.4f})"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {SECONDARY_COLOR};'>📊 지역 간 상대 비교</h4>", unsafe_allow_html=True)
        st.plotly_chart(viz['compare'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {SECONDARY_COLOR};'>🍩 인프라 카테고리 비중</h4>", unsafe_allow_html=True)
        st.plotly_chart(viz['pie'], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3층: 상세 트리맵
    if 'tree' in viz:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {SECONDARY_COLOR}; margin-bottom: 1rem;'>🌳 인프라 상세 구성 트리맵</h4>", unsafe_allow_html=True)
        st.plotly_chart(viz['tree'], use_container_width=True)
        st.caption("거리 기반 가중치가 적용된 시설 분포도입니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 하단 데이터 테이블 (익스팬더)
    with st.expander("📍 분석 반경 내 상세 시설 목록 확인하기"):
        if facilities:
            st.dataframe(pd.DataFrame(facilities)[['group', 'sub_category', 'name', 'distance', 'emoji']], 
                         use_container_width=True, height=400)
        else:
            st.warning("분석 반경 내에 해당하는 시설이 발견되지 않았습니다.")

# 푸터
st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 4rem 0 2rem 0; font-size: 0.9rem; border-top: 1px solid #e2e8f0;">
        <p>© 2026 Seoul Seulsekwon Analytics. Empowered by Public Open Data.</p>
        <p style="margin-top: 0.5rem; font-size: 0.8rem;">Kakao Local API & Streamlit Framework | Designed for Data Analysts</p>
    </div>
""", unsafe_allow_html=True)
