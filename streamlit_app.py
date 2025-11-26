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

# --- APP CONFIGURATION & STYLING ---
st.set_page_config(page_title="Fishmouth Pro", page_icon="🐟", layout="centered")

# Inject custom CSS for an "Industrial Blueprint" look
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #0e3c61 !important; /* Dark Industrial Blue */
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Stylish Expanders (for settings) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #d1d9e6;
        border-radius: 5px;
        color: #0e3c61;
        font-weight: bold;
    }
    
    /* Tab Styling (Blueprint Look) */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #dfe7f0; /* Light blue-gray */
        color: #0e3c61 !important;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        border: 1px solid #c0cddb;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #d93025 !important; /* Accent Red for selected */
        border-top: 3px solid #d93025;
    }
    
    /* Metrics and Data */
    [data-testid="stMetricLabel"] { font-size: 0.9rem; color: #555; }
    [data-testid="stMetricValue"] { color: #0e3c61; }
    
    /* Custom Header */
    .main-header {
        padding: 1rem;
        background: linear-gradient(90deg, #0e3c61 0%, #1a5f96 100%);
        color: white;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Custom Branded Header
st.markdown('<div class="main-header"><h1>🐟 Fishmouth Pro</h1><p style="margin-bottom:0;">Industrial Pipe Fabrication Calculator</p></div>', unsafe_allow_html=True)

# --- NAVIGATION Menu ---
# Use radio buttons horizontally for a cleaner mobile nav look
tool_mode = st.radio("Select Tool:", ["🐟 Fishmouth (Tees/Laterals)", "🦞 Lobster Back (Elbows)", "📐 Miter Master"], horizontal=True, label_visibility="collapsed")
st.divider()

# ==============================================================================
# TOOL 1: THE FISHMOUTH
# ==============================================================================
if "Fishmouth" in tool_mode:
    # --- INPUTS ---
    with st.expander("⚙️ **Pipe Geometry Settings** (Tap to Open)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Header Pipe")
            header_nom = st.selectbox("Nominal Size", list(pipe_schedule.keys()), index=12, key="h_size")
            header_od = pipe_schedule[header_nom]
            header_type = st.radio("Shape", ["Straight", "Elbow"], horizontal=True)
            st.caption(f"O.D.: {header_od}\"")
        with col2:
            st.markdown("##### Branch Pipe")
            branch_nom = st.selectbox("Nominal Size", list(pipe_schedule.keys()), index=10, key="b_size")
            branch_od = pipe_schedule[branch_nom]
            st.caption(f"O.D.: {branch_od}\"")
        
        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            angle = st.number_input("Angle (°)", 1.0, 90.0, 90.0)
        with col4:
            max_off = max(0.0, (header_od - branch_od)/2)
            offset = st.slider("Eccentric Offset", 0.0, max_off, 0.0, step=0.125)

    # --- CALCULATION ---
    if branch_od > header_od:
        st.error("🚨 **CRITICAL ERROR:** Branch pipe is larger than Header pipe.")
    else:
        R, r = header_od/2, branch_od/2
        if r >= R and offset == 0: r = R - 0.001 # Lie Rule
        
        theta = np.linspace(0, 2*np.pi, 33)
        alpha = np.radians(angle)
        
        x = theta * r
        # Simplified math for display
        term_sq = R**2 - (r*np.sin(theta) + offset)**2
        term_sq[term_sq < 0] = 0
        if angle == 90: y = np.sqrt(term_sq)
        else: y = (np.sqrt(term_sq)/np.sin(alpha)) + (r*np.cos(theta)/np.tan(alpha))
        y = y - np.min(y)

        # --- OUTPUT TABS ---
        st.write("") # Spacer
        tab1, tab2 = st.tabs(["📉 BLUEPRINT VIEW", "🔢 ORDINATE DATA"])
        
        with tab1:
            st.write("##### Cut Template")
            fig, ax = plt.subplots(figsize=(6, 3))
            # Style the plot to look like a blueprint
            fig.patch.set_facecolor('#f0f2f6')
            ax.set_facecolor('#ffffff')
            ax.plot(x, y, color='#0e3c61', lw=2.5)
            ax.fill_between(x, y, color='#d9eaf7', alpha=0.5)
            ax.set_xlim(0, max(x))
            ax.set_ylim(0, max(y)*1.15)
            ax.set_xlabel("Circumference (inches)", fontweight='bold', color='#0e3c61')
            ax.set_ylabel("Measure UP (inches)", fontweight='bold', color='#0e3c61')
            ax.grid(True, which='both', color='#c0cddb', linestyle='--', alpha=0.7)
            for i in range(0, 33, 4):
                ax.vlines(x[i], 0, y[i], color='#0e3c61', lw=0.5, linestyle=':')
                ax.text(x[i], -0.15*max(y), f"{int(i/2)+1}", ha='center', fontsize=9, color='#d93025', fontweight='bold')
            st.pyplot(fig)
            if offset > 0: st.warning("👉 **Eccentric Cut:** Align line #1 with the closest side of the header pipe.")

        with tab2:
            st.write("##### Measurement Table")
            indices = range(0, 33, 2)
            df = pd.DataFrame({
                "Line #": [int(i/2)+1 for i in indices],
                "Decimal Inc.": [round(y[i], 3) for i in indices],
                "Approx Fraction": [f"{int(y[i])} {int((y[i]%1)*16)}/16" for i in indices]
            })
            st.dataframe(df, hide_index=True, use_container_width=True, height=400)

# ==============================================================================
# TOOL 2: LOBSTER BACK
# ==============================================================================
elif "Lobster" in tool_mode:
    st.write("### 🦞 Segmented Elbow Calculator")
    st.caption("Calculates middle and end segments for multi-piece bends.")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            pipe_nom = st.selectbox("Pipe Size", list(pipe_schedule.keys()), index=12)
            pipe_od = pipe_schedule[pipe_nom]
        with col2:
            pieces = st.selectbox("Total Pieces", [3, 4, 5, 6], index=1)
        
        col3, col4 = st.columns(2)
        with col3:
            bend_angle = st.number_input("Total Bend Angle", 1, 180, 90)
        with col4:
            default_rad = 1.5 * float(eval(pipe_nom.replace("-", "+")))
            radius = st.number_input("Bend Radius (Centerline)", value=default_rad)

    num_welds = pieces - 1
    miter_angle = bend_angle / (num_welds * 2)
    middle_spine = 2 * np.tan(np.radians(miter_angle)) * (radius + pipe_od/2)
    middle_throat = 2 * np.tan(np.radians(miter_angle)) * (radius - pipe_od/2)

    st.write("")
    st.subheader("Fabrication Dimensions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Miter Angle (Cut)", f"{round(miter_angle, 1)}°")
    c2.metric("Middle Long Side", f"{round(middle_spine, 3)}\"", help="Length of the longest part of the segment")
    c3.metric("Middle Short Side", f"{round(middle_throat, 3)}\"", help="Length of the shortest part of the segment")

    st.write("##### Middle Segment Diagram")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#f0f2f6')
    # Draw Trapezoid
    p = patches.Polygon([[0, 0], [middle_spine, 0], [middle_spine - (middle_spine-middle_throat)/2, pipe_od], [(middle_spine-middle_throat)/2, pipe_od]], closed=True, fill=True, facecolor='#d9eaf7', edgecolor='#0e3c61', lw=2)
    ax.add_patch(p)
    ax.set_xlim(-0.5, middle_spine+0.5)
    ax.set_ylim(-0.5, pipe_od+1)
    ax.set_aspect('equal')
    ax.axis('off')
    # Dimensions
    ax.annotate('', xy=(0, -0.2), xytext=(middle_spine, -0.2), arrowprops=dict(arrowstyle='<->', color='#0e3c61'))
    ax.text(middle_spine/2, -0.5, f"Long: {round(middle_spine, 3)}\"", ha='center', color='#0e3c61', fontweight='bold')
    ax.annotate('', xy=((middle_spine-middle_throat)/2, pipe_od+0.2), xytext=(middle_spine - (middle_spine-middle_throat)/2, pipe_od+0.2), arrowprops=dict(arrowstyle='<->', color='#0e3c61'))
    ax.text(middle_spine/2, pipe_od+0.5, f"Short: {round(middle_throat, 3)}\"", ha='center', color='#0e3c61')
    st.pyplot(fig)
    
    st.info(f"**Cut List:**\n- **{pieces-2}** Middle pieces (dimensions above).\n- **2** End pieces (one square end, one {round(miter_angle, 1)}° miter end).")

# ==============================================================================
# TOOL 3: MITER MASTER
# ==============================================================================
elif "Miter" in tool_mode:
    st.write("### 📐 Simple Miter Calculator")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            pipe_nom = st.selectbox("Pipe Size", list(pipe_schedule.keys()), index=8)
            pipe_od = pipe_schedule[pipe_nom]
        with col2:
            cut_angle = st.number_input("Desired Cut Angle", 1.0, 89.0, 45.0)

    cut_height = np.tan(np.radians(cut_angle)) * pipe_od
    
    st.write("")
    st.metric("Cutback Measurement", f"{round(cut_height, 3)}\"", help="Distance from the short point to the long point.")
    
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor('#f0f2f6')
    ax.plot([0, pipe_od], [0, 0], color='#0e3c61', lw=2) # Bottom
    ax.plot([0, pipe_od], [cut_height, 0], color='#d93025', lw=3) # Cut line
    ax.plot([0, 0], [0, cut_height], color='#0e3c61', linestyle='--') # Height
    ax.text(-0.1, cut_height/2, f"{round(cut_height, 3)}\"", ha='right', color='#d93025', fontweight='bold')
    ax.set_xlim(-1, pipe_od+0.5)
    ax.set_ylim(-0.5, cut_height+0.5)
    ax.axis('off')
    st.pyplot(fig)
    
    st.success("Mark the pipe circumference. Measure the 'Cutback' distance from that line to find your long point. Connect the dots.")
