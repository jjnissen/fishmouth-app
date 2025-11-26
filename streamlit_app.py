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

# --- SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'tool' not in st.session_state: st.session_state.tool = None

def reset():
    st.session_state.step = 1
    st.session_state.tool = None

# --- STYLING (The "Pretty" Part) ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        width: 100%; height: 55px; font-weight: bold; border-radius: 8px; border: 1px solid #d1d9e6;
    }
    .instruction-box {
        background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px; margin-bottom: 20px;
    }
    .big-text { font-size: 20px; font-weight: bold; color: #0d47a1; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: DRAW INSTRUCTIONAL DIAGRAMS ---
def draw_concept_visual(mode, h_od, b_od, offset=0):
    """Draws a simple helper diagram to explain concepts to the user"""
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_aspect('equal')
    ax.axis('off')
    
    if mode == "ECCENTRIC":
        # Draw Header Circle (Big)
        header = plt.Circle((0, 0), h_od/2, color='#e0e0e0', label='Header')
        ax.add_patch(header)
        # Draw Branch Circle (Small)
        branch = plt.Circle((0, offset), b_od/2, color='#2196f3', alpha=0.7, label='Branch')
        ax.add_patch(branch)
        
        ax.set_xlim(-h_od/1.5, h_od/1.5)
        ax.set_ylim(-h_od/1.5, h_od/1.5)
        ax.text(0, -h_od/1.8, "End View: Offset", ha='center', fontsize=8)
        
    elif mode == "ELBOW_VS_STRAIGHT":
        # Draw Straight Pipe
        ax.plot([-2, 2], [0, 0], color='gray', lw=20, alpha=0.3)
        ax.text(0, -0.5, "Straight Header", ha='center')
        
    return fig

# ==============================================================================
# STEP 1: HOME
# ==============================================================================
if st.session_state.step == 1:
    st.title("🐟 Fishmouth Pro")
    st.markdown("### Select Fabrication Type")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🐟 Tee / Lateral"):
            st.session_state.tool = "Fishmouth"
            st.session_state.step = 2
            st.rerun()
        if st.button("🦞 Segmented Elbow"):
            st.session_state.tool = "Lobster"
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("📐 Simple Miter"):
            st.session_state.tool = "Miter"
            st.session_state.step = 2
            st.rerun()
        if st.button("Y  True Wye"):
            st.session_state.tool = "Wye"
            st.session_state.step = 2
            st.rerun()

# ==============================================================================
# STEP 2: GUIDED SETUP
# ==============================================================================
elif st.session_state.step == 2:
    if st.button("← Back"): reset(); st.rerun()
    
    if st.session_state.tool == "Fishmouth":
        st.markdown('<p class="big-text">1. Select Pipe Sizes</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            h_nom = st.selectbox("Header (Main Pipe)", all_sizes, index=12)
        with c2:
            # Smart Logic: Branch cannot be larger than header
            valid_branches = all_sizes[:all_sizes.index(h_nom)+1]
            b_nom = st.selectbox("Branch (Top Pipe)", valid_branches, index=len(valid_branches)-1)
            
        h_od, b_od = pipe_schedule[h_nom], pipe_schedule[b_nom]

        st.markdown('<p class="big-text">2. Geometry & Offset</p>', unsafe_allow_html=True)
        h_type = st.radio("Header Shape", ["Straight Pipe", "Elbow Fitting"], horizontal=True)
        angle = st.number_input("Intersection Angle (°)", 1.0, 90.0, 90.0)

        # --- THE SMART SLIDER LOGIC ---
        max_off = max(0.0, (h_od - b_od)/2)
        
        # We draw the instructional diagram LIVE
        c_draw, c_input = st.columns([1, 2])
        
        offset = 0.0
        with c_input:
            if max_off <= 0.001:
                st.warning("🔒 Offset Locked: Branch is same size as Header.")
            else:
                offset = st.slider("Eccentric Offset (Inches)", 0.0, max_off, 0.0, step=0.125)
                if offset > 0:
                    st.caption("Moves the branch off-center (See diagram).")

        with c_draw:
            # Draw the live diagram showing the circles moving
            st.pyplot(draw_concept_visual("ECCENTRIC", h_od, b_od, offset))

        if st.button("🚀 Generate Template"):
            st.session_state.inputs = {"h_nom": h_nom, "b_nom": b_nom, "h_type": h_type, "angle": angle, "offset": offset, "h_od": h_od, "b_od": b_od}
            st.session_state.step = 3
            st.rerun()

    # (Other tools abbreviated for brevity, logic remains same as previous version)
    elif st.session_state.tool == "Lobster":
         # ... [Keep previous Lobster logic] ...
         st.markdown('<p class="big-text">🦞 Lobster Setup</p>', unsafe_allow_html=True)
         p_nom = st.selectbox("Pipe Size", all_sizes, index=12)
         pieces = st.selectbox("Pieces", [3, 4, 5, 6], index=1)
         bend = st.number_input("Total Angle", 90)
         rad = st.number_input("Radius", value=1.5 * float(eval(p_nom.replace("-", "+"))))
         if st.button("Calculate"):
             st.session_state.inputs = {"p_nom": p_nom, "pieces": pieces, "bend": bend, "rad": rad}
             st.session_state.step = 3
             st.rerun()
    
    elif st.session_state.tool in ["Miter", "Wye"]:
         # ... [Keep previous Miter logic] ...
         st.markdown('<p class="big-text">📐 Miter Setup</p>', unsafe_allow_html=True)
         p_nom = st.selectbox("Pipe Size", all_sizes, index=8)
         angle = st.number_input("Angle", 45.0)
         if st.button("Calculate"):
             st.session_state.inputs = {"p_nom": p_nom, "angle": angle}
             st.session_state.step = 3
             st.rerun()


# ==============================================================================
# STEP 3: HIGH QUALITY RESULTS
# ==============================================================================
elif st.session_state.step == 3:
    if st.button("← Start Over"): reset(); st.rerun()
    
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

        # --- GUIDANCE BOX ---
        st.markdown(f"""
        <div class="instruction-box">
            <b>📋 Job Instructions:</b><br>
            1. Mark a <b>Base Line</b> ring around your {d['b_nom']}" pipe.<br>
            2. Divide the Base Line into <b>16 equal spaces</b>.<br>
            3. Measure UP from the Base Line using the numbers below.
        </div>
        """, unsafe_allow_html=True)

        # --- THE "PRETTY" PLOT ---
        st.markdown("### ✂️ Cutting Template")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 1. Fill the "Waste" area (The part you cut off)
        ax.fill_between(x, y, max(y)*1.1, color='#ffebee', label="Waste Material")
        # 2. Fill the "Keep" area (The Pipe)
        ax.fill_between(x, 0, y, color='#e3f2fd', label="Pipe Template")
        # 3. Draw the Cut Line
        ax.plot(x, y, color='#d32f2f', lw=3, label="CUT LINE")
        
        # 4. Draw Base Line
        ax.axhline(0, color='black', lw=2)
        ax.text(-0.5, 0, "Base Line", ha='right', va='center', fontweight='bold')

        # 5. Draw Arrows (Like the Book)
        for i in range(0, 33, 4):
            ax.annotate('', xy=(x[i], y[i]), xytext=(x[i], 0), arrowprops=dict(arrowstyle='->', color='black'))
            ax.text(x[i], -0.1*max(y), f"{int(i/2)+1}", ha='center', fontsize=9, fontweight='bold', color='#1565c0')

        ax.set_xlim(-1, max(x)+1)
        ax.set_ylim(-0.2*max(y), max(y)*1.2)
        ax.axis('off')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        st.pyplot(fig)
        
        # --- DATA TABLE ---
        st.markdown("### 📏 Measures")
        df = pd.DataFrame({
            "Line #": [int(i/2)+1 for i in range(0, 33, 2)],
            "Decimal": [round(y[i], 3) for i in range(0, 33, 2)],
            "Fraction": [f"{int(y[i])} {int((y[i]%1)*16)}/16" for i in range(0, 33, 2)]
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

    # (Lobster/Miter result display code is same as previous, just keep it clean)
    elif st.session_state.tool == "Lobster":
        # ... [Reuse Math] ...
        d = st.session_state.inputs
        p_od = pipe_schedule[d['p_nom']]
        num_welds = d['pieces'] - 1
        miter_angle = d['bend'] / (num_welds * 2)
        long = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] + p_od/2)
        short = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] - p_od/2)
        
        st.markdown(f"""
        <div class="instruction-box">
            <b>🦞 Lobster Instructions:</b><br>
            Cut <b>{d['pieces']-2} middle pieces</b> using the dimensions below.<br>
            Cut <b>2 end pieces</b> (one side square, one side mitered).
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Angle", f"{round(miter_angle, 1)}°")
        c2.metric("Long Side", f"{round(long, 3)}\"")
        c3.metric("Short Side", f"{round(short, 3)}\"")
        
        # Simple Visual
        fig, ax = plt.subplots(figsize=(6, 2.5))
        pts = [[0, 0], [long, 0], [long - (long-short)/2, p_od], [(long-short)/2, p_od]]
        p = patches.Polygon(pts, closed=True, fill=True, facecolor='#e3f2fd', edgecolor='black')
        ax.add_patch(p)
        ax.set_xlim(-1, long+1); ax.set_ylim(-1, p_od+1); ax.axis('off')
        ax.text(long/2, -0.4, "Long Side", ha='center', fontweight='bold')
        ax.text(long/2, p_od+0.3, "Short Side", ha='center', fontweight='bold')
        st.pyplot(fig)

    elif st.session_state.tool in ["Miter", "Wye"]:
        d = st.session_state.inputs
        p_od = pipe_schedule[d['p_nom']]
        cut_height = np.tan(np.radians(d['angle'])) * p_od
        st.metric("Cutback", f"{round(cut_height, 3)}\"")
        
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot([0, p_od], [0, 0], 'k-', lw=2)
        ax.plot([0, p_od], [cut_height, 0], 'r-', lw=4, label='Cut Line')
        ax.plot([0, 0], [0, cut_height], 'k--', lw=1)
        ax.text(-0.2, cut_height/2, f"{round(cut_height, 3)}\"", ha='right', color='red', fontweight='bold')
        ax.set_xlim(-1, p_od+1); ax.set_ylim(-0.5, cut_height+0.5); ax.axis('off')
        st.pyplot(fig)
