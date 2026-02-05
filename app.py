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

load_dotenv()

st.set_page_config(page_title="서울 슬세권 지수 대시보드", page_icon="🏙️", layout="wide")

PRIMARY_COLOR = "#3b82f6"
SECONDARY_COLOR = "#1e293b"
ACCENT_COLOR = "#6366f1"
BACKGROUND_COLOR = "#f8fafc"
CARD_BG = "#ffffff"

EMOJI_MAP = {
    "스타벅스": "☕", "편의점": "🏪", "세탁소": "🏪", "마트": "🏪", "대형마트": "🏬",
    "백화점": "🏬", "버스": "🚌", "bus": "🚌", "정류장": "🚌", "정류소": "🚌",
    "지하철": "🚇", "metro": "🚇", "역": "🚇", "병원": "🏥", "의원": "💊",
    "약국": "💊", "경찰": "🚓", "파출소": "🚓", "도서관": "📚", "서점": "📚",
    "학교": "🏫", "공원": "🌳", "park": "🌳", "체육": "🏋️", "운동": "🏋️", "은행": "🏦", "금융": "🏦"
}

CATEGORY_GROUPS = {
    "생활/편의🏪": ["스타벅스", "편의점", "세탁소", "마트", "대형마트", "백화점", "카페"],
    "교통🚌": ["버스", "지하철", "정류장", "정류소", "역", "bus", "metro"],
    "의료💊": ["병원", "의원", "약국", "치과", "한의원"],
    "안전/치안🚨": ["경찰", "파출소", "소방"],
    "교육/문화📚": ["도서관", "서점", "학교", "유치원", "학원"],
    "자연/여가🌳": ["공원", "체육", "운동", "산책", "park"],
    "금융🏦": ["은행", "금융", "ATM"]
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp {{ font-family: 'Inter', sans-serif; background-color: {BACKGROUND_COLOR}; }}
    .dashboard-card {{
        background: {CARD_BG}; padding: 1.5rem; border-radius: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.8); margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }}
    .dashboard-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 28px -5px rgba(0, 0, 0, 0.08); }}
    .metric-value {{
        font-size: 5rem; font-weight: 800;
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }}
    .grade-badge-s {{ background-color: #f59e0b; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; }}
    .grade-badge-a {{ background-color: #10b981; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; }}
    .grade-badge-b {{ background-color: #3b82f6; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; }}
    .grade-badge-c {{ background-color: #64748b; color: white; padding: 0.6rem 1.8rem; border-radius: 9999px; font-weight: 700; font-size: 1.3rem; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 분석 엔진 로직
# ==========================================

def get_coords_from_address(address: str):
    api_key = None
    try:
        if "KAKAO_REST_API_KEY" in st.secrets: api_key = st.secrets["KAKAO_REST_API_KEY"]
    except: pass
    if not api_key: api_key = os.getenv("KAKAO_REST_API_KEY")
    if not api_key: return None
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    try:
        response = requests.get(url, headers=headers, params={"query": address})
        if response.status_code == 200:
            result = response.json()
            if result['documents']:
                info = result['documents'][0]
                return {"address_name": info['address_name'], "lat": float(info['y']), "lng": float(info['x'])}
    except: pass
    return None

def get_dong_name(address):
    if not isinstance(address, str): return "알 수 없음"
    match = re.search(r'([가-힣]+동)', address)
    return match.group(1) if match else "서울시 전체"

@st.cache_data
def load_all_data():
    base_path = "data/cleaned"
    if not os.path.exists(base_path): base_path = os.path.join(os.path.dirname(__file__), "data/cleaned")
    if not os.path.exists(base_path): return pd.DataFrame()

    file_map = {
        'starbucks_seoul_cleaned.csv': '스타벅스', 'bus_station_seoul_cleaned.csv': '버스정류장',
        'metro_station_seoul_cleaned.csv': '지하철역', 'hospital_seoul_cleaned.csv': '병원',
        'police_seoul_cleaned_ver2.csv': '경찰서', 'library_seoul_cleaned.csv': '도서관',
        'bookstore_seoul_cleaned.csv': '서점', 'school_seoul_cleaned.csv': '학교',
        'park_raw_cleaned_revised.csv': '공원', 'finance_seoul_cleaned.csv': '은행',
        'large_scale_shop_seoul_cleaned.csv': '대형마트', 'sosang_seoul_cleaned.csv': '소상공인',
        'sosang_seoul_cleaned_ver2.csv': '소상공인'
    }

    all_dfs = []
    # 매우 강력한 컬럼 매핑
    lat_names = ['위도', 'lat', 'latitude', '좌표정보(Y)', 'Y', 'y', 'lat_wgs84', '위도(WGS84)']
    lon_names = ['경도', 'lon', 'longitude', 'lng', '좌표정보(X)', 'X', 'x', 'lon_wgs84', '경도(WGS84)']
    name_names = ['상호명', '점포명', '정류소명', '이름', '사업장명', '시설명', '공원명', '도서관명', '학교명', '기관명', 'name']

    for file, default_cat in file_map.items():
        path = os.path.join(base_path, file)
        if os.path.exists(path):
            df = None
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except: continue
            
            if df is not None:
                # 서브 카테고리 결정 로직 강화 (NaN 처리 포함)
                if '카테고리_소' in df.columns:
                    df['sub_category'] = df['카테고리_소'].fillna(default_cat)
                elif '업태구분명' in df.columns:
                    df['sub_category'] = df['업태구분명'].fillna(default_cat)
                else:
                    df['sub_category'] = default_cat
                
                # 빈 문자열 처리
                df['sub_category'] = df['sub_category'].replace('', default_cat)
                
                lat_c = next((c for c in lat_names if c in df.columns), None)
                lon_c = next((c for c in lon_names if c in df.columns), None)
                name_c = next((c for c in name_names if c in df.columns), None)

                if lat_c and lon_c:
                    if not name_c: 
                        name_c = next((c for c in df.columns if any(k in str(c) for k in ['명', '이름', '역', '정류'])), df.columns[0])
                    
                    temp_df = df[[name_c, lat_c, lon_c, 'sub_category']].copy()
                    temp_df.columns = ['name', 'lat', 'lon', 'sub_category']
                    temp_df['lat'] = pd.to_numeric(temp_df['lat'], errors='coerce')
                    temp_df['lon'] = pd.to_numeric(temp_df['lon'], errors='coerce')
                    temp_df = temp_df.dropna(subset=['lat', 'lon'])
                    
                    # 좌표 필터링 범위 최적화 및 이상치 제거 (위도 37~38, 경도 126~128 사이 집중)
                    mask = (temp_df['lat'] > 36.0) & (temp_df['lat'] < 39.0) & \
                           (temp_df['lon'] > 125.0) & (temp_df['lon'] < 129.0)
                    temp_df = temp_df[mask]
                    
                    if not temp_df.empty:
                        all_dfs.append(temp_df)
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def calculate_seulsekwon_index(center_lat, center_lon, data, weights, radius_m):
    if data.empty: return 0.0, {cat: 0.0 for cat in CATEGORY_GROUPS.keys()}, {cat: 0 for cat in CATEGORY_GROUPS.keys()}, [], {cat: 0.0 for cat in CATEGORY_GROUPS.keys()}
    radius_km = radius_m / 1000.0
    # 기준치 현실화 (도심 내 500m 반경 기준)
    max_counts = {"생활/편의🏪": 15, "교통🚌": 8, "의료💊": 5, "안전/치안🚨": 1, "교육/문화📚": 2, "자연/여가🌳": 2, "금융🏦": 3}
    
    lat_margin, lon_margin = radius_km / 111.0, radius_km / 88.0
    mask = (data['lat'] >= center_lat - lat_margin) & (data['lat'] <= center_lat + lat_margin) & \
           (data['lon'] >= center_lon - lon_margin) & (data['lon'] <= center_lon + lon_margin)
    filtered = data[mask].copy()

    scores, counts, nearby, raw_scores = {}, {}, [], {}
    for g_name, sub_cats in CATEGORY_GROUPS.items():
        # 대소문자 무시 및 부분 일치 검색 강화
        g_data = filtered[filtered['sub_category'].apply(lambda x: any(str(sc).lower() in str(x).lower() for sc in sub_cats))]
        actual_count = 0
        for _, row in g_data.iterrows():
            dist = geodesic((center_lat, center_lon), (row['lat'], row['lon'])).meters
            if dist <= radius_m:
                actual_count += 1
                r_dict = row.to_dict(); r_dict['distance'] = dist; r_dict['group'] = g_name
                r_dict['emoji'] = next((emoji for key, emoji in EMOJI_MAP.items() if key in str(row['sub_category'])), "📍")
                nearby.append(r_dict)
        counts[g_name] = actual_count
        m = max_counts.get(g_name, 5)
        rate = min(actual_count, m) / m
        raw_scores[g_name] = rate
        scores[g_name] = round(rate * weights.get(g_name, 0), 2)
    
    # 가까운 시설 우선 표시를 위해 거리순 정렬
    nearby = sorted(nearby, key=lambda x: x['distance'])
    
    total = round(sum(scores.values()), 1)
    return total, scores, counts, nearby, raw_scores

def create_visualizations(total_score, scores, counts, facilities, dong_name, raw_scores):
    layout_opts = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color=SECONDARY_COLOR))
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[v * 100 for v in raw_scores.values()] + [list(raw_scores.values())[0] * 100],
        theta=list(raw_scores.keys()) + [list(raw_scores.keys())[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',
        line=dict(color=ACCENT_COLOR, width=3),
        name='카테고리 달성률'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0%", "25%", "50%", "75%", "100%"]
            )
        ),
        showlegend=False,
        **layout_opts
    )
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=total_score, title={'text': "슬세권 종합 지수"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': ACCENT_COLOR},
               'steps': [{'range': [0, 70], 'color': "#fee2e2"}, {'range': [70, 85], 'color': "#fef3c7"}, {'range': [85, 100], 'color': "#dcfce7"}]}
    ))
    fig_gauge.update_layout(height=300, **layout_opts)
    
    fig_compare = px.bar(x=[f"'{dong_name}'", "서울 평균"], y=[total_score, 75.5], color=[f"'{dong_name}'", "서울 평균"],
                         color_discrete_map={f"'{dong_name}'": PRIMARY_COLOR, "서울 평균": "#cbd5e1"})
    fig_compare.update_layout(showlegend=False, height=300, **layout_opts)
    
    fig_pie = px.pie(names=list(counts.keys()), values=list(counts.values()), hole=.6)
    fig_pie.update_layout(height=300, showlegend=True, legend=dict(orientation="h", y=-0.2), **layout_opts)
    
    return {'radar': fig_radar, 'gauge': fig_gauge, 'compare': fig_compare, 'pie': fig_pie}

def create_enhanced_map(lat, lon, facilities, radius_m):
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles="cartodbpositron")
    folium.Circle([lat, lon], radius=radius_m, color=PRIMARY_COLOR, fill=True, fill_opacity=0.1).add_to(m)
    folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='home', prefix='fa'), tooltip="분석 지점").add_to(m)
    # 지도에 표시할 시설물 개수 상향 (최대 500개) 및 카테고리별 분산 배치
    for f in facilities[:500]:
        html = f'<div style="font-size: 16px; background: white; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2); border: 2.5px solid {ACCENT_COLOR};">{f["emoji"]}</div>'
        folium.Marker([f['lat'], f['lon']], icon=folium.DivIcon(html=html), popup=f"<b>{f['name']}</b><br>{f['distance']:.0f}m").add_to(m)
    return m

# ==========================================
# 3. Streamlit UI 메인
# ==========================================

if 'data' not in st.session_state:
    with st.status("🚀 분석 엔진 및 지도 데이터 초기화 중...", expanded=True) as status:
        st.write("📊 대용량 지리 정보 데이터를 로드하고 있습니다...")
        st.session_state.data = load_all_data()
        if not st.session_state.data.empty:
            st.write(f"✅ 총 {len(st.session_state.data):,}개의 생활 인프라 데이터 추출 완료")
            status.update(label="분석 준비 완료", state="complete", expanded=False)
        else:
            st.error("🚨 데이터를 로드하지 못했습니다. /data/cleaned 폴더를 확인하세요.")
            status.update(label="초기화 실패", state="error", expanded=True)

# 초기 세션 상태 설정
state_init = {
    'coords': (37.5006, 127.0363), 'address': "역삼역", 'radius': 500,
    'weights': {"생활/편의🏪": 30, "교통🚌": 20, "의료💊": 15, "안전/치안🚨": 10, "교육/문화📚": 5, "자연/여가🌳": 15, "금융🏦": 5}
}
for k, v in state_init.items():
    if k not in st.session_state: st.session_state[k] = v

st.markdown(f'<h1 style="text-align: center; color: {SECONDARY_COLOR}; margin-bottom: 2rem;">🏙️ SEOUL SEULSEKWON DASHBOARD</h1>', unsafe_allow_html=True)

# 검색 섹션
with st.form("main_search"):
    c1, c2, c3 = st.columns([2.5, 1, 1])
    with c1: query = st.text_input("📍 분석할 주소 또는 건물명", value=st.session_state.address, placeholder="예: 강남역, 한남동 6-1")
    with c2: rad = st.select_slider("📏 분석 반경 (m)", options=[300, 500, 700, 1000, 1500], value=st.session_state.radius)
    with c3: st.write("<div style='height:28px;'></div>", unsafe_allow_html=True); submit = st.form_submit_button("실시간 지수 분석하기")

if submit and query:
    with st.spinner("🔍 해당 위치의 데이터셋을 매핑하는 중..."):
        res = get_coords_from_address(query)
        if res:
            st.session_state.coords = (res['lat'], res['lng'])
            st.session_state.address = res['address_name']
            st.session_state.radius = rad; st.rerun()
        else: st.error("❌ 위치 정보를 찾을 수 없습니다. 주소를 정확히 입력해 주세요.")

if st.session_state.address:
    with st.sidebar:
        st.header("⚖️ 인프라 가중치 조정")
        with st.form("custom_weights"):
            new_w = {cat: st.slider(cat, 0, 50, val) for cat, val in st.session_state.weights.items()}
            if st.form_submit_button("가중치 즉각 적용"):
                st.session_state.weights = new_w; st.rerun()
        st.success(f"데이터셋: {len(st.session_state.data):,}건 로드됨")
        if st.button("🔄 엔진 재부팅 (캐시 삭제)"):
            st.cache_data.clear(); st.rerun()

    t_score, scores, counts, facilities, raw_scores = calculate_seulsekwon_index(
        st.session_state.coords[0], st.session_state.coords[1], st.session_state.data, st.session_state.weights, st.session_state.radius
    )
    dong = get_dong_name(st.session_state.address)
    viz = create_visualizations(t_score, scores, counts, facilities, dong, raw_scores)

    # 지수 카드 및 주요 차트
    c1, c2, c3 = st.columns([1.1, 1, 0.9])
    with c1:
        grade = "s" if t_score >= 90 else ("a" if t_score >= 80 else ("b" if t_score >= 70 else "c"))
        st.markdown(f'<div class="dashboard-card score-container"><p style="color:#64748b; font-size:0.9rem;">{st.session_state.address} 분석 결과</p><div class="metric-value">{t_score}</div><span class="grade-badge-{grade}">{grade.upper()} GRADE</span></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="dashboard-card">', unsafe_allow_html=True); st.plotly_chart(viz['radar'], use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="dashboard-card">', unsafe_allow_html=True); st.plotly_chart(viz['gauge'], use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)

    # 지도 및 세부 통계
    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True); st.subheader("🗺️ 시설물 상세 분포 지도")
        m = create_enhanced_map(st.session_state.coords[0], st.session_state.coords[1], facilities, st.session_state.radius)
        map_interaction = st_folium(m, width="100%", height=500, key="main_map")
        if map_interaction and map_interaction.get("last_clicked"):
            nc = (map_interaction["last_clicked"]["lat"], map_interaction["last_clicked"]["lng"])
            if round(nc[0], 5) != round(st.session_state.coords[0], 5):
                st.session_state.coords = nc; st.session_state.address = f"지정 포인트 ({nc[0]:.4f}, {nc[1]:.4f})"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True); st.subheader("📊 인프라 밸런스"); st.plotly_chart(viz['compare'], use_container_width=True); st.plotly_chart(viz['pie'], use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📍 반경 내 전체 시설 현황 (검색 가능)"):
        if facilities: st.dataframe(pd.DataFrame(facilities)[['group', 'sub_category', 'name', 'distance', 'emoji']], use_container_width=True)
        else: st.info("분석 반경 내에 해당하는 시설 정보가 없습니다.")
st.markdown("<div style='text-align: center; color: #94a3b8; padding: 2rem;'>© 2026 Seoul Seulsekwon Analytics Engine v2.5</div>", unsafe_allow_html=True)
