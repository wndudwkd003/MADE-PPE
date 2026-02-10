import streamlit as st
import json
import time
import os
import re
from pathlib import Path

# 페이지 기본 설정
st.set_page_config(
    page_title="PPE Log Replay",
    layout="wide",
    page_icon="👷"
)

# --------------------------------------------------------------------------
# 0. 설정 및 상수
# --------------------------------------------------------------------------

# 허용할 폴더 패턴 및 표시 이름 매핑
AGENT_MAPPING = {
    r"^MADE\d+$": "MADE-PPE",
    r"^SINGLE_ONESHOT\d+$": "Single (One-Pass)",
    r"^SINGLE_STEP\d+$": "Single (Multi-Step)"
}

TARGET_DATASET = "SCP300_a"

# --------------------------------------------------------------------------
# 1. 데이터 로더 유틸리티
# --------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_friendly_name(folder_name):
    """폴더 이름을 UI 표시용 이름으로 변환 (정규표현식 사용)"""
    for pattern, name in AGENT_MAPPING.items():
        if re.match(pattern, folder_name):
            # 예: SINGLE_ONESHOT1 -> Single (One-Pass) #1
            suffix = folder_name.replace(folder_name.rstrip('0123456789'), '')
            return f"{name} #{suffix}" if suffix else name
    return None

def get_runs_structure(base_dir="runs"):
    """
    구조: runs/SCP300/{AGENT_VER}/split/outputs/all/*.json
    """
    structure = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        return {}

    # 1. Dataset (SCP300만 허용)
    dataset_path = base_path / TARGET_DATASET
    if not dataset_path.exists():
        return {}

    structure[TARGET_DATASET] = {}

    # 2. Agent Version (MADE1, SINGLE_ONESHOT1 등)
    for agent_dir in dataset_path.iterdir():
        if agent_dir.is_dir():
            friendly_name = get_friendly_name(agent_dir.name)

            # 허용된 패턴이 아니면 스킵
            if not friendly_name:
                continue

            # UI 표시용 키 생성 (예: "Single (One-Pass) #1")
            # 실제 경로를 찾기 위해 튜플이나 별도 매핑이 필요하지만,
            # 여기서는 간단히 구조체에 실제 폴더명을 저장하고 UI에서 변환

            # 3. Split (train/test/valid)
            for split in agent_dir.iterdir():
                 if split.is_dir() and (split / "outputs/all").exists():
                     if friendly_name not in structure[TARGET_DATASET]:
                         structure[TARGET_DATASET][friendly_name] = {}

                     # 실제 폴더명 저장
                     if agent_dir.name not in structure[TARGET_DATASET][friendly_name]:
                         structure[TARGET_DATASET][friendly_name][agent_dir.name] = []

                     structure[TARGET_DATASET][friendly_name][agent_dir.name].append(split.name)
    return structure

# --------------------------------------------------------------------------
# 2. 로그 파싱 및 변환 로직 (핵심 수정 사항)
# --------------------------------------------------------------------------
def parse_logs_for_display(logs):
    """
    서로 다른 로그 구조(OneShot, MultiStep, MADE)를
    통일된 '표시용 로그 리스트'로 변환합니다.
    """
    display_logs = []

    if not logs:
        return []

    # Case A: One-Shot (로그가 1개이고 stage가 ONESHOT인 경우)
    if logs[0].get("stage") == "ONESHOT":
        log = logs[0]
        parsed = log.get("response", {}).get("parsed", {})
        latency = log.get("latency_sec", 0)

        # One-Shot 결과를 5단계로 가상의 로그 분할
        steps = [
            ("Work Environment", parsed.get("work_environment")),
            ("Hazard", parsed.get("hazard")),
            ("Compliance", parsed.get("compliance")),
            ("Wearing", parsed.get("wearing")),
            ("Improper Wearing", parsed.get("improper_wearing"))
        ]

        for step_name, step_data in steps:
            if step_data:
                display_logs.append({
                    "stage": "ONESHOT ANALYSIS",
                    "role": "Single Agent",
                    "round": 1,
                    "header": step_name, # 소제목
                    "text": step_data.get("reason", ""),
                    "json_data": step_data,
                    "latency": latency / 5  # 시간은 단순히 나눔 (시각화용)
                })

    # Case B: Multi-Step / MADE (여러 로그가 존재하는 경우)
    else:
        for log in logs:
            stage = log.get("stage", "UNKNOWN")
            role = log.get("role", "Agent") # Role이 없으면 Agent로 통일
            round_idx = log.get("round", 1)
            latency = log.get("latency_sec", 0)

            parsed = log.get("response", {}).get("parsed", {})

            text_content = ""
            json_content = None

            # 텍스트/JSON 추출 로직
            if isinstance(parsed, dict):
                if "text" in parsed:
                    text_content = parsed["text"]
                    # 추가 필드가 있으면 JSON으로 저장
                    if len(parsed) > 1:
                        json_content = {k:v for k,v in parsed.items() if k != "text"}
                elif "reason" in parsed:
                     text_content = parsed["reason"]
                     json_content = parsed
                else:
                    text_content = "Structured Output Generated"
                    json_content = parsed
            elif isinstance(parsed, str):
                text_content = parsed

            display_logs.append({
                "stage": stage,
                "role": role,
                "round": round_idx,
                "header": None,
                "text": text_content,
                "json_data": json_content,
                "latency": latency
            })

    return display_logs

# --------------------------------------------------------------------------
# 3. 사이드바: 파일 탐색기
# --------------------------------------------------------------------------
st.sidebar.title("🗂️ Log Explorer")

runs_data = get_runs_structure()

if not runs_data:
    st.error(f"No valid data found in 'runs/{TARGET_DATASET}'. Check directory structure.")
    st.stop()

# 1. Dataset (SCP300 고정)
selected_dataset = TARGET_DATASET

# 2. Agent Name (Friendly Name)
friendly_agents = list(runs_data[selected_dataset].keys())
selected_friendly_agent = st.sidebar.selectbox("Agent Model", friendly_agents)

# 실제 폴더명 찾기 (내부 구조에서)
real_agent_folder = None
if selected_friendly_agent:
    # 구조: {friendly: {real_folder: [splits]}}
    agent_map = runs_data[selected_dataset][selected_friendly_agent]
    real_agent_folder = list(agent_map.keys())[0] # 보통 하나만 매핑됨
    splits = agent_map[real_agent_folder]

    # 3. Split
    selected_split = st.sidebar.selectbox("Split", splits)
else:
    selected_split = None

# 4. JSON File
selected_file_path = None
if selected_split and real_agent_folder:
    json_dir = Path("runs") / selected_dataset / real_agent_folder / selected_split / "outputs" / "all"

    if json_dir.exists():
        json_files = sorted(list(json_dir.glob("*.json")))
        file_names = [f.name for f in json_files]

        selected_filename = st.sidebar.selectbox(
            "Select Image Log",
            file_names,
            index=0
        )
        selected_file_path = json_dir / selected_filename
    else:
        st.sidebar.warning("No output directory found.")

# Replay 설정
st.sidebar.markdown("---")
st.sidebar.header("🎬 Replay Settings")
replay_mode = st.sidebar.checkbox("Enable Animation Effect", value=True)
replay_speed = st.sidebar.slider("Speed (sec/msg)", 0.1, 2.0, 0.5)


# --------------------------------------------------------------------------
# 4. 메인 화면: 시각화
# --------------------------------------------------------------------------
if selected_file_path:
    data = load_json(selected_file_path)

    # 헤더
    st.title(f"🛡️ Analysis Replay: {selected_friendly_agent}")
    st.caption(f"File: {selected_file_path.name}")

    col1, col2 = st.columns([1, 1.5])

    # [왼쪽] 이미지 및 최종 결과
    with col1:
        st.subheader("🖼️ Target Image")
        img_path = data.get("image", "")

        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_path}")
            st.image("https://via.placeholder.com/400x300?text=No+Image", use_container_width=True)

        st.divider()
        st.subheader("🏁 Final Decision")

        # 최종 결과 JSON
        final_labels = data.get("labels", {})
        with st.expander("View Final Labels", expanded=True):
            st.json(final_labels)

    # [오른쪽] 토론 로그 (변환된 로그 사용)
    with col2:
        st.subheader("💬 Inference Process")
        chat_container = st.container()

        raw_logs = data.get("logs", [])

        # 로그 변환 (OneShot 분해, 텍스트 추출 등)
        display_logs = parse_logs_for_display(raw_logs)

        if not display_logs:
            st.info("No logs found.")

        with chat_container:
            current_stage = None

            for log in display_logs:
                stage = log["stage"]
                role = log["role"]
                header = log["header"] # OneShot용 소제목
                text = log["text"]
                json_data = log["json_data"]
                latency = log["latency"]

                # 스테이지 구분선
                if stage != current_stage:
                    st.markdown(f"#### 🚩 Stage: {stage}")
                    st.divider()
                    current_stage = stage
                    if replay_mode: time.sleep(0.3)

                # 아바타 설정
                avatar = "🤖"
                if "PROPOSER" in role.upper(): avatar = "🙋‍♂️"
                elif "REBUTTER" in role.upper(): avatar = "🤔"
                elif "JUDGE" in role.upper(): avatar = "⚖️"
                elif "SINGLE" in role.upper() or "AGENT" in role.upper(): avatar = "🧠"

                # 채팅 메시지 출력
                with st.chat_message(name=role, avatar=avatar):
                    # 상단 캡션 (소제목이 있으면 표시)
                    caption_text = f"**{role}**"
                    if header:
                        caption_text += f" | Step: {header}"
                    caption_text += f" ({latency:.2f}s)"
                    st.caption(caption_text)

                    # 본문 텍스트 (마침표 줄바꿈 처리)
                    if text:
                        formatted_text = text.replace(".", ".\n\n")
                        st.markdown(formatted_text)

                    # JSON 데이터 (접이식)
                    if json_data:
                        with st.expander("🔍 Structured Data"):
                            st.json(json_data)

                # 애니메이션 딜레이
                if replay_mode:
                    time.sleep(replay_speed)

        if replay_mode:
            st.success("Analysis Finished!")

else:
    st.info("👈 Please select a log file from the sidebar.")
