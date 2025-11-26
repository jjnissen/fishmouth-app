import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. THE DATABASE ---
pipe_schedule = {
    "1/8": 0.405, "1/4": 0.540, "3/8": 0.675, "1/2": 0.840,
    "3/4": 1.050, "1": 1.315, "1-1/4": 1.660, "1-1/2": 1.900,
    "2": 2.375, "2-1/2": 2.875, "3": 3.500, "3-1/2": 4.000,
    "4": 4.500, "5": 5.563, "6": 6.625, "8": 8.625,
    "10": 10.750, "12": 12.750, "14": 14.000, "16": 16.000,
    "18": 18.000, "20": 20.000, "24": 24.000
}

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Fishmouth Pro", page_icon="🐟")

st.title("🐟 Fishmouth Pro")
st.write("Industrial Pipe Fabrication Calculator")

# --- NAVIGATION ---
# Using standard tabs for navigation which is much safer on mobile
nav_mode = st.tabs(["🐟 Fishmouth", "🦞 Lobster Back", "📐 Miter Cut"])

# ==============================================================================
# TAB 1: THE FISHMOUTH
# ==============================================================================
with nav_mode[0]:
    st.header("Tee & Lateral Calculator")
    
    # --- INPUTS ---
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            header_nom = st.selectbox("Header Size", list(pipe_schedule.keys()), index=12, key="h_size")
            header_od = pipe_schedule[header_nom]
            header_type = st.radio("Header Shape", ["Straight", "Elbow"], horizontal=True)
            st.caption(f"O.D.: {header_od}\"")
        with col2:
            branch_nom = st.selectbox("Branch Size", list(pipe_schedule.keys()), index=10, key="b_size")
            branch_od = pipe_schedule[branch_nom]
            st.caption(f"O.D.: {branch_od}\"")
        
        col3, col4 = st.columns(2)
        with col3:
            angle = st.number_input("Angle (°)", 1.0, 90.0, 90.0)
        with col4:
            max_off = max(0.0, (header_od - branch_od)/2)
            offset = st.slider("Eccentric Offset", 0.0, max_off, 0.0, step=0.125)

    # --- CALCULATION ---
    if branch_od > header_od:
        st.error("🚨 ERROR: Branch pipe cannot be larger than Header pipe.")
    else:
        R, r = header_od/2, branch_od/2
        if r >= R and offset == 0: r = R - 0.001 # Lie Rule
        
        theta = np.linspace(0, 2*np.pi, 33)
        alpha = np.radians(angle)
        
        x = theta * r
        term_sq = R**2 - (r*np.sin(theta) + offset)**2
        term_sq[term_sq < 0] = 0
        if angle == 90: y = np.sqrt(term_sq)
        else: y = (np.sqrt(term_sq)/np.sin(alpha)) + (r*np.cos(theta)/np.tan(alpha))
        y = y - np.min(y)

        # --- OUTPUT ---
        st.divider()
        st.subheader("1. Template Pattern")
        
        # Graph
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, color='black', lw=2)
        ax.fill_between(x, y, color='gray', alpha=0.2)
        ax.set_xlim(0, max(x))
        ax.set_ylim(0, max(y)*1.2)
        ax.set_xlabel("Circumference (inches)")
        ax.set_ylabel("Measure UP (inches)")
        ax.grid(True, linestyle=':')
        st.pyplot(fig)
        
        if offset > 0: st.info("Mark 'Left/Right' Hand as per Page 40.")

        st.subheader("2. Measurement Table")
        indices = range(0, 33, 2)
        df = pd.DataFrame({
            "Line #": [int(i/2)+1 for i in indices],
            "Inches": [round(y[i], 3) for i in indices],
            "Fraction": [f"{int(y[i])} {int((y[i]%1)*16)}/16" for i in indices]
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 2: LOBSTER BACK
# ==============================================================================
with nav_mode[1]:
    st.header("Segmented Elbow Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        pipe_nom = st.selectbox("Pipe Size", list(pipe_schedule.keys()), index=12, key="l_size")
        pipe_od = pipe_schedule[pipe_nom]
    with col2:
        pieces = st.selectbox("Total Pieces", [3, 4, 5, 6], index=1)
    
    col3, col4 = st.columns(2)
    with col3:
        bend_angle = st.number_input("Total Bend Angle", 1, 180, 90)
    with col4:
        default_rad = 1.5 * float(eval(pipe_nom.replace("-", "+")))
        radius = st.number_input("Bend Radius", value=default_rad)

    num_welds = pieces - 1
    miter_angle = bend_angle / (num_welds * 2)
    middle_spine = 2 * np.tan(np.radians(miter_angle)) * (radius + pipe_od/2)
    middle_throat = 2 * np.tan(np.radians(miter_angle)) * (radius - pipe_od/2)

    st.success(f"**Cut Angle:** {round(miter_angle, 1)}°")
    
    c1, c2 = st.columns(2)
    c1.metric("Long Side", f"{round(middle_spine, 3)}\"")
    c2.metric("Short Side", f"{round(middle_throat, 3)}\"")
    
    st.write("---")
    st.write("**Visual Reference (Middle Piece):**")
    
    fig, ax = plt.subplots(figsize=(6, 2))
    p = patches.Polygon([[0, 0], [middle_spine, 0], [middle_spine - (middle_spine-middle_throat)/2, pipe_od], [(middle_spine-middle_throat)/2, pipe_od]], closed=True, fill=True, facecolor='#eeeeee', edgecolor='black')
    ax.add_patch(p)
    ax.set_xlim(-0.5, middle_spine+0.5)
    ax.set_ylim(-0.5, pipe_od+1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(middle_spine/2, -0.4, "Long Side", ha='center')
    ax.text(middle_spine/2, pipe_od+0.2, "Short Side", ha='center')
    st.pyplot(fig)

# ==============================================================================
# TAB 3: MITER MASTER
# ==============================================================================
with nav_mode[2]:
    st.header("Simple Miter Cut")
    
    col1, col2 = st.columns(2)
    with col1:
        pipe_nom = st.selectbox("Pipe Size", list(pipe_schedule.keys()), index=8, key="m_size")
        pipe_od = pipe_schedule[pipe_nom]
    with col2:
        cut_angle = st.number_input("Desired Cut Angle", 1.0, 89.0, 45.0)

    cut_height = np.tan(np.radians(cut_angle)) * pipe_od
    
    st.metric("Cutback Measurement", f"{round(cut_height, 3)}\"")
    st.caption("Measure this distance from the cut line.")
    
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, pipe_od], [0, 0], color='black')
    ax.plot([0, pipe_od], [cut_height, 0], color='red', lw=3)
    ax.plot([0, 0], [0, cut_height], color='black', linestyle='--')
    ax.text(-0.1, cut_height/2, f"{round(cut_height, 3)}\"", ha='right', color='red')
    ax.set_xlim(-1, pipe_od+0.5)
    ax.set_ylim(-0.5, cut_height+0.5)
    ax.axis('off')
    st.pyplot(fig)
