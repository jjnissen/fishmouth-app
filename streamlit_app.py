import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. THE DATABASE (Standard Pipe Schedules) ---
pipe_schedule = {
    "1/8": 0.405, "1/4": 0.540, "3/8": 0.675, "1/2": 0.840,
    "3/4": 1.050, "1": 1.315, "1-1/4": 1.660, "1-1/2": 1.900,
    "2": 2.375, "2-1/2": 2.875, "3": 3.500, "3-1/2": 4.000,
    "4": 4.500, "5": 5.563, "6": 6.625, "8": 8.625,
    "10": 10.750, "12": 12.750, "14": 14.000, "16": 16.000,
    "18": 18.000, "20": 20.000, "24": 24.000
}

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Fishmouth Pro", page_icon="🐟", layout="centered")

# --- CSS FOR MOBILE OPTIMIZATION ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b; }
    h1 { font-size: 1.8rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🐟 Fishmouth Pro")

# --- NAVIGATION ---
tool_mode = st.selectbox("Select Tool:", ["🐟 Fishmouth (Tees/Laterals)", "🦞 Lobster Back (Elbows)", "📐 Miter Master"], label_visibility="collapsed")

# ==============================================================================
# TOOL 1: THE FISHMOUTH (Tees, Laterals, Eccentric)
# ==============================================================================
if "Fishmouth" in tool_mode:
    st.caption("Generate Cutting Templates for Pipe Intersections")
    
    # --- INPUTS ---
    with st.expander("⚙️ Pipe Settings (Tap to Open)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            header_nom = st.selectbox("Header Size", list(pipe_schedule.keys()), index=12)
            header_od = pipe_schedule[header_nom]
            header_type = st.radio("Header Shape", ["Straight", "Elbow"], horizontal=True)
        with col2:
            branch_nom = st.selectbox("Branch Size", list(pipe_schedule.keys()), index=10)
            branch_od = pipe_schedule[branch_nom]
        
        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            angle = st.number_input("Angle (°)", 1.0, 90.0, 90.0)
        with col4:
            # Smart Slider for Offset
            max_off = max(0.0, (header_od - branch_od)/2)
            offset = st.slider("Eccentric Offset", 0.0, max_off, 0.0, step=0.125)

    # --- CALCULATION ---
    if branch_od > header_od:
        st.error("❌ Branch cannot be larger than Header")
    else:
        # Lie Rule Logic
        R, r = header_od/2, branch_od/2
        if r >= R and offset == 0: r = R - 0.001
        
        theta = np.linspace(0, 2*np.pi, 33)
        alpha = np.radians(angle)
        
        # Math Engine
        x = theta * r
        if header_type == "Elbow":
            # Simplified Elbow Logic (Page 42 approx)
            term_sq = R**2 - (r*np.sin(theta) + offset)**2
        else:
            term_sq = R**2 - (r*np.sin(theta) + offset)**2
        
        term_sq[term_sq < 0] = 0
        
        if angle == 90:
            y = np.sqrt(term_sq)
        else:
            y = (np.sqrt(term_sq)/np.sin(alpha)) + (r*np.cos(theta)/np.tan(alpha))
        
        y = y - np.min(y)

        # --- OUTPUT TABS ---
        tab1, tab2 = st.tabs(["📉 Template Graph", "🔢 The Numbers"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(x, y, 'k-', lw=2)
            ax.fill_between(x, y, color='skyblue', alpha=0.3)
            ax.set_xlim(0, max(x))
            ax.set_ylim(0, max(y)*1.2)
            ax.set_xlabel("Circumference (inches)")
            ax.set_ylabel("Measure UP (inches)")
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
            # Draw Ordinate Lines
            for i in range(0, 33, 2):
                ax.vlines(x[i], 0, y[i], color='gray', lw=0.5)
                # Only label a few to keep mobile clean
                if i % 4 == 0:
                    ax.text(x[i], -0.1*max(y), f"{int(i/2)+1}", ha='center', fontsize=8, color='blue')
            
            st.pyplot(fig)
            if offset > 0:
                st.info("ℹ️ Eccentric Cut: Align lowest point with closest side of header.")

        with tab2:
            # Data Table
            indices = range(0, 33, 2)
            df = pd.DataFrame({
                "Line #": [int(i/2)+1 for i in indices],
                "Inches": [round(y[i], 3) for i in indices],
                "Fraction (Approx)": [f"{int(y[i])} {int((y[i]%1)*16)}/16" for i in indices]
            })
            st.dataframe(df, hide_index=True, use_container_width=True)

# ==============================================================================
# TOOL 2: THE LOBSTER BACK (Multi-Piece Elbows)
# ==============================================================================
elif "Lobster" in tool_mode:
    st.caption("Calculate Segmented Elbows (Page 58)")
    
    col1, col2 = st.columns(2)
    with col1:
        pipe_nom = st.selectbox("Pipe Size", list(pipe_schedule.keys()), index=12)
        pipe_od = pipe_schedule[pipe_nom]
    with col2:
        pieces = st.selectbox("Number of Pieces", [3, 4, 5, 6], index=1) # Default 4
    
    col3, col4 = st.columns(2)
    with col3:
        bend_angle = st.number_input("Bend Angle", 1, 180, 90)
    with col4:
        # Default Radius is 1.5x Diameter (Long Radius)
        default_rad = 1.5 * float(eval(pipe_nom.replace("-", "+")))
        radius = st.number_input("Bend Radius", value=default_rad)

    # --- LOBSTER MATH ---
    # Formula: Miter Angle = Total Angle / ((Pieces - 1) * 2)
    num_welds = pieces - 1
    miter_angle = bend_angle / (num_welds * 2)
    tangent_len = np.tan(np.radians(miter_angle)) * (pipe_od / 2)
    
    # Middle Segment Lengths
    middle_spine_len = 2 * np.tan(np.radians(miter_angle)) * (radius + pipe_od/2)
    middle_throat_len = 2 * np.tan(np.radians(miter_angle)) * (radius - pipe_od/2)
    
    # End Piece Lengths (User Defined usually, but we give min required)
    min_end = tangent_len + 1.0 # Add 1 inch for safety

    st.divider()
    
    # --- VISUAL RESULTS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Cut Angle", f"{round(miter_angle, 1)}°")
    c2.metric("Spine (Long)", f"{round(middle_spine_len, 2)}\"")
    c3.metric("Throat (Short)", f"{round(middle_throat_len, 2)}\"")

    st.subheader("Construction Diagram")
    
    # Draw a simplified segment diagram
    fig, ax = plt.subplots(figsize=(6, 2))
    # Draw Trapazoid representing a segment
    p = patches.Polygon([
        [0, 0], 
        [middle_spine_len, 0], 
        [middle_spine_len - (middle_spine_len-middle_throat_len)/2, pipe_od], 
        [(middle_spine_len-middle_throat_len)/2, pipe_od]
    ], closed=True, fill=True, facecolor='#e0f7fa', edgecolor='black')
    ax.add_patch(p)
    ax.set_xlim(-1, middle_spine_len+1)
    ax.set_ylim(-1, pipe_od+1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(middle_spine_len/2, -0.5, f"Long Side: {round(middle_spine_len, 3)}\"", ha='center')
    ax.text(middle_spine_len/2, pipe_od + 0.2, f"Short Side: {round(middle_throat_len, 3)}\"", ha='center')
    ax.set_title(f"Middle Segment Dimensions (Qty: {pieces-2})")
    st.pyplot(fig)
    
    st.info(f"**Instructions:** Cut {pieces-2} middle pieces using the dimensions above. Cut 2 End pieces with one square end and one {round(miter_angle, 1)}° miter.")

# ==============================================================================
# TOOL 3: MITER MASTER (Any Angle Cuts)
# ==============================================================================
elif "Miter" in tool_mode:
    st.caption("Calculate Any-Angle Miter Cuts (Page 50)")
    
    col1, col2 = st.columns(2)
    with col1:
        pipe_nom = st.selectbox("Pipe Size", list(pipe_schedule.keys()), index=8)
        pipe_od = pipe_schedule[pipe_nom]
    with col2:
        cut_angle = st.number_input("Cut Angle", 1.0, 89.0, 45.0)

    # Simple Miter Math
    # Height of cut = tan(angle) * Diameter
    cut_height = np.tan(np.radians(cut_angle)) * pipe_od
    
    st.metric("Cutback Measurement", f"{round(cut_height, 3)}\"")
    
    # Visual
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, pipe_od], [0, 0], 'k-') # Bottom
    ax.plot([0, pipe_od], [cut_height, 0], 'r-', lw=3) # Cut line
    ax.plot([0, 0], [0, cut_height], 'k--') # Height
    ax.text(-0.2, cut_height/2, f"{round(cut_height, 3)}\"", ha='right', color='red', fontweight='bold')
    ax.set_xlim(-1, pipe_od+1)
    ax.set_ylim(-0.5, cut_height+0.5)
    ax.axis('off')
    st.pyplot(fig)
    
    st.success("Use this 'Cutback' measure to mark the long point from the short point on your pipe.")
