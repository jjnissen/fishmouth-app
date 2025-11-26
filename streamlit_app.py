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
all_sizes = list(pipe_schedule.keys())

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Fishmouth Pro", page_icon="🐟")

# --- SESSION STATE (Memory for the Wizard) ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'tool' not in st.session_state: st.session_state.tool = None

def reset():
    st.session_state.step = 1
    st.session_state.tool = None

# --- STYLING ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    .big-header { font-size: 24px; font-weight: bold; color: #0e3c61; margin-bottom: 20px;}
    .result-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #d1d9e6; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# STEP 1: WHAT ARE WE BUILDING?
# ==============================================================================
if st.session_state.step == 1:
    st.title("🐟 Fishmouth Pro")
    st.write("Select the cut you need to make:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🐟 Tee / Lateral"):
            st.session_state.tool = "Fishmouth"
            st.session_state.step = 2
            st.rerun()
        if st.button("🦞 Segmented Elbow"):
            st.session_state.tool = "Lobster"
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("📐 Simple Miter"):
            st.session_state.tool = "Miter"
            st.session_state.step = 2
            st.rerun()
        if st.button("Y  True Wye"):
            st.session_state.tool = "Wye"
            st.session_state.step = 2
            st.rerun()

# ==============================================================================
# STEP 2: THE QUESTIONS (Wizard)
# ==============================================================================
elif st.session_state.step == 2:
    if st.button("← Back"): reset(); st.rerun()
    
    # --- FISHMOUTH QUESTIONS ---
    if st.session_state.tool == "Fishmouth":
        st.markdown('<p class="big-header">🐟 Tee & Lateral Setup</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            h_nom = st.selectbox("Header Pipe Size", all_sizes, index=12)
        with c2:
            # Smart Filter
            valid_branches = all_sizes[:all_sizes.index(h_nom)+1]
            b_nom = st.selectbox("Branch Pipe Size", valid_branches, index=len(valid_branches)-1)
        
        h_type = st.radio("Is the Header Straight or an Elbow?", ["Straight", "Elbow"], horizontal=True)
        angle = st.number_input("Intersection Angle (°)", 1.0, 90.0, 90.0)
        
        # Smart Slider
        h_od, b_od = pipe_schedule[h_nom], pipe_schedule[b_nom]
        max_off = max(0.0, (h_od - b_od)/2)
        offset = 0.0
        if max_off > 0.01:
            offset = st.slider("Offset (Eccentric)", 0.0, max_off, 0.0, step=0.125)
            
        if st.button("🚀 Calculate Cut"):
            st.session_state.inputs = {
                "h_nom": h_nom, "b_nom": b_nom, "h_type": h_type, "angle": angle, "offset": offset,
                "h_od": h_od, "b_od": b_od
            }
            st.session_state.step = 3
            st.rerun()

    # --- LOBSTER QUESTIONS ---
    elif st.session_state.tool == "Lobster":
        st.markdown('<p class="big-header">🦞 Segmented Elbow Setup</p>', unsafe_allow_html=True)
        p_nom = st.selectbox("Pipe Size", all_sizes, index=12)
        pieces = st.selectbox("How many pieces?", [3, 4, 5, 6], index=1)
        bend = st.number_input("Total Bend Angle", 1, 180, 90)
        rad = st.number_input("Bend Radius", value=1.5 * float(eval(p_nom.replace("-", "+"))))
        
        if st.button("🚀 Calculate Segments"):
            st.session_state.inputs = {"p_nom": p_nom, "pieces": pieces, "bend": bend, "rad": rad}
            st.session_state.step = 3
            st.rerun()

    # --- MITER/WYE QUESTIONS ---
    elif st.session_state.tool in ["Miter", "Wye"]:
        label = "Simple Miter" if st.session_state.tool == "Miter" else "True Wye"
        st.markdown(f'<p class="big-header">📐 {label} Setup</p>', unsafe_allow_html=True)
        
        p_nom = st.selectbox("Pipe Size", all_sizes, index=8)
        
        if st.session_state.tool == "Wye":
            st.info("For a standard 90° Wye, enter 45° below.")
            
        angle = st.number_input("Cut Angle", 1.0, 89.0, 45.0)
        
        if st.button("🚀 Calculate Cut"):
            st.session_state.inputs = {"p_nom": p_nom, "angle": angle}
            st.session_state.step = 3
            st.rerun()

# ==============================================================================
# STEP 3: THE RESULTS (The "Magic")
# ==============================================================================
elif st.session_state.step == 3:
    if st.button("← Start Over"): reset(); st.rerun()
    
    # --- FISHMOUTH RESULTS ---
    if st.session_state.tool == "Fishmouth":
        d = st.session_state.inputs
        
        # MATH
        R, r = d['h_od']/2, d['b_od']/2
        if r >= R and d['offset'] == 0: r = R - 0.001
        
        theta = np.linspace(0, 2*np.pi, 33)
        alpha = np.radians(d['angle'])
        x = theta * r
        
        term = R**2 - (r*np.sin(theta) + d['offset'])**2
        term[term<0] = 0
        if d['angle'] == 90: y = np.sqrt(term)
        else: y = (np.sqrt(term)/np.sin(alpha)) + (r*np.cos(theta)/np.tan(alpha))
        y = y - np.min(y)

        # 1. THE LAYOUT (PAGE 74 AUTOMATION)
        st.markdown("### 1. Header Layout (Locating the Pipe)")
        
        # Calculate distance from center line (Page 74 Math)
        # Distance = (Header OD / 2) * cot(angle)
        # cot(a) = 1/tan(a)
        if d['angle'] != 90:
            layout_dist = (d['h_od'] / 2) / np.tan(alpha)
            direction = "Forward" if layout_dist > 0 else "Backward"
            
            st.info(f"""
            **Marking the Header Pipe:**
            1. Draw a Center Line on the Header.
            2. Measure **{round(abs(layout_dist), 3)}"** ({direction}) from that center line.
            3. This is your **Throat (Point B)** intersection point.
            """)
        else:
            st.success("Angle is 90°. The Branch fits directly on the Center Line.")

        # 2. THE TEMPLATE
        st.markdown("### 2. Branch Template (The Cut)")
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, 'k-', lw=2)
        ax.fill_between(x, y, color='#d9eaf7', alpha=0.5)
        ax.set_xlim(0, max(x))
        ax.set_ylim(0, max(y)*1.2)
        ax.grid(True, linestyle=':')
        ax.set_xlabel("Circumference")
        st.pyplot(fig)
        
        # 3. THE DATA
        st.markdown("### 3. Measures (From Base Line)")
        df = pd.DataFrame({
            "Line": [int(i/2)+1 for i in range(0, 33, 2)],
            "Inches": [round(y[i], 3) for i in range(0, 33, 2)],
            "Fraction": [f"{int(y[i])} {int((y[i]%1)*16)}/16" for i in range(0, 33, 2)]
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

    # --- LOBSTER RESULTS ---
    elif st.session_state.tool == "Lobster":
        d = st.session_state.inputs
        p_od = pipe_schedule[d['p_nom']]
        
        num_welds = d['pieces'] - 1
        miter_angle = d['bend'] / (num_welds * 2)
        long = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] + p_od/2)
        short = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] - p_od/2)
        
        st.markdown("### Fabrication Specs")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cut Angle", f"{round(miter_angle, 1)}°")
        c2.metric("Long Side", f"{round(long, 3)}\"")
        c3.metric("Short Side", f"{round(short, 3)}\"")
        
        st.info(f"Cut **{d['pieces']-2}** middle pieces using these dimensions.")
        
        # Draw Diagram
        fig, ax = plt.subplots(figsize=(6, 2))
        p = patches.Polygon([[0, 0], [long, 0], [long - (long-short)/2, p_od], [(long-short)/2, p_od]], closed=True, fill=True, facecolor='#eeeeee', edgecolor='black')
        ax.add_patch(p)
        ax.set_xlim(-0.5, long+0.5)
        ax.set_ylim(-0.5, p_od+1)
        ax.axis('off')
        ax.text(long/2, -0.3, "Long Side", ha='center')
        ax.text(long/2, p_od+0.2, "Short Side", ha='center')
        st.pyplot(fig)

    # --- MITER RESULTS ---
    elif st.session_state.tool in ["Miter", "Wye"]:
        d = st.session_state.inputs
        p_od = pipe_schedule[d['p_nom']]
        cut_height = np.tan(np.radians(d['angle'])) * p_od
        
        st.markdown("### Cutback Layout")
        st.metric("Cutback Distance", f"{round(cut_height, 3)}\"")
        st.caption("Mark the circumference. Measure this distance to find the long point.")
        
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot([0, p_od], [0, 0], 'k-')
        ax.plot([0, p_od], [cut_height, 0], 'r-', lw=3)
        ax.plot([0, 0], [0, cut_height], 'k--')
        ax.text(-0.1, cut_height/2, f"{round(cut_height, 3)}\"", ha='right', color='red', fontweight='bold')
        ax.set_xlim(-1, p_od+0.5)
        ax.set_ylim(-0.5, cut_height+0.5)
        ax.axis('off')
        st.pyplot(fig)
