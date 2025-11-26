import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

st.set_page_config(page_title="Fishmouth Pro", page_icon="🐟")

if 'step' not in st.session_state: st.session_state.step = 1
if 'tool' not in st.session_state: st.session_state.tool = None

def reset():
    st.session_state.step = 1
    st.session_state.tool = None

# --- STYLING ---
st.markdown("""
    <style>
    div.stButton > button:first-child { 
        width: 100%; height: 60px; font-weight: bold; border-radius: 10px; 
        border: 1px solid #d1d9e6; font-size: 18px;
    }
    .hero-box {
        background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 1px solid #b0c4de; margin-bottom: 20px;
    }
    .instruction-box { 
        background-color: #fff; border-left: 5px solid #2196f3; padding: 15px; margin-bottom: 20px; color: #333; 
        border: 1px solid #eee;
    }
    .step-header { font-size: 24px; font-weight: bold; color: #0e3c61; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- VISUAL HELPERS ---
def draw_concept_visual(mode, h_od, b_od, offset=0):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_aspect('equal'); ax.axis('off')
    if mode == "ECCENTRIC":
        ax.add_patch(plt.Circle((0, 0), h_od/2, facecolor='white', edgecolor='#0e3c61', lw=2))
        ax.add_patch(plt.Circle((0, offset), b_od/2, facecolor='#d9eaf7', edgecolor='#2196f3', lw=2, alpha=0.9))
        if offset > 0: ax.annotate('', xy=(0, 0), xytext=(0, offset), arrowprops=dict(arrowstyle='<-', color='red', lw=2))
        ax.set_xlim(-h_od/1.4, h_od/1.4); ax.set_ylim(-h_od/1.4, h_od/1.4)
    return fig

def draw_markup_guide():
    """Visualizes the Base Line Process"""
    fig, ax = plt.subplots(figsize=(6, 3))
    # Draw Pipe
    ax.add_patch(patches.Rectangle((0, 0), 6, 3, linewidth=2, edgecolor='#0e3c61', facecolor='white'))
    
    # Draw Base Line
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5)
    ax.text(3, 0.2, "1. DRAW BASE LINE", ha='center', fontsize=10, fontweight='bold')
    
    # Draw Measurement lines
    x_pos = [1, 2, 3, 4, 5]
    heights = [1.0, 2.0, 2.5, 2.0, 1.0]
    
    for i, x in enumerate(x_pos):
        ax.vlines(x, 0.5, 0.5 + heights[i], color='#2196f3', lw=1)
        ax.plot(x, 0.5 + heights[i], 'ro', markersize=5) # The Dot
        if i == 2:
            ax.text(x + 0.1, 1.5, "2. MEASURE UP", color='#2196f3', fontweight='bold')
            
    # Connect dots
    ax.plot(x_pos, [h + 0.5 for h in heights], color='#b71c1c', linestyle=':', linewidth=2)
    ax.text(4, 3.2, "3. CONNECT DOTS", ha='center', color='#b71c1c', fontweight='bold')
    
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(0, 3.5); ax.axis('off')
    return fig

def plot_overlay_on_image(bg_image, x_vals, y_vals, scale, x_shift, y_shift):
    dpi = 100; height, width = np.array(bg_image).shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(bg_image)
    x_center = np.mean(x_vals)
    x_scaled = (x_vals - x_center) * scale + (width / 2) + x_shift
    y_scaled = (height / 2) - (y_vals * scale) + y_shift
    ax.plot(x_scaled, y_scaled, color='#ff0000', linewidth=5, alpha=0.8) 
    base_y = (height / 2) + y_shift
    ax.axhline(base_y, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
    ax.axis('off')
    return fig

# ==============================================================================
# STEP 1: HOME (THE "WHY" & "1-2-3")
# ==============================================================================
if st.session_state.step == 1:
    st.title("🐟 Fishmouth Pro")
    
    st.markdown("""
    <div class="hero-box">
        <b>Welcome to precision fabrication.</b><br>
        Stop guessing. Stop grinding. This app calculates the exact geometry to make pipes fit perfectly every time.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("#### 1. Select your Cut:")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🐟 Tee / Lateral"): 
            st.session_state.tool = "Fishmouth"; st.session_state.step = 2; st.rerun()
        st.caption("**Best for:** Branching a pipe off a main header. Precise fit for welding.")
        
        if st.button("🦞 Segmented Elbow"): 
            st.session_state.tool = "Lobster"; st.session_state.step = 2; st.rerun()
        st.caption("**Best for:** Turning corners with custom radiuses when fittings aren't available.")
        
    with c2:
        if st.button("📐 Simple Miter"): 
            st.session_state.tool = "Miter"; st.session_state.step = 2; st.rerun()
        st.caption("**Best for:** Fast 45° or 90° turns on handrails or frames.")
        
        if st.button("Y  True Wye"): 
            st.session_state.tool = "Wye"; st.session_state.step = 2; st.rerun()
        st.caption("**Best for:** Splitting flow into two equal pipes.")

# ==============================================================================
# STEP 2: MEASURE (THE QUESTIONS)
# ==============================================================================
elif st.session_state.step == 2:
    if st.button("← Back"): reset(); st.rerun()
    st.markdown('<p class="step-header">2. Measure & Input</p>', unsafe_allow_html=True)
    st.progress(50)
    
    if st.session_state.tool == "Fishmouth":
        c1, c2 = st.columns(2)
        with c1: h_nom = st.selectbox("Header Size", all_sizes, index=12)
        with c2: b_nom = st.selectbox("Branch Size", all_sizes[:all_sizes.index(h_nom)+1], index=len(all_sizes[:all_sizes.index(h_nom)+1])-1)
        h_od, b_od = pipe_schedule[h_nom], pipe_schedule[b_nom]

        st.write("**Geometry:**")
        h_type = st.radio("Header Shape", ["Straight Pipe", "Elbow Fitting"], horizontal=True)
        angle = st.number_input("Intersection Angle (°)", 1.0, 90.0, 90.0)

        max_off = max(0.0, (h_od - b_od)/2)
        c_draw, c_input = st.columns([1, 1.5])
        offset = 0.0
        with c_input:
            if max_off <= 0.001:
                st.slider("Offset (Locked)", 0.0, 1.0, 0.0, disabled=True)
                st.caption("🔒 Branch equals Header size.")
            else:
                offset = st.slider("Eccentric Offset", 0.0, max_off, 0.0, step=0.125)
        with c_draw:
            st.pyplot(draw_concept_visual("ECCENTRIC", h_od, b_od, offset))

        st.divider()
        if st.button("🚀 Get Markings"):
            st.session_state.inputs = {"h_nom": h_nom, "b_nom": b_nom, "h_type": h_type, "angle": angle, "offset": offset, "h_od": h_od, "b_od": b_od}
            st.session_state.step = 3; st.rerun()

    elif st.session_state.tool == "Lobster":
         p_nom = st.selectbox("Pipe Size", all_sizes, index=12); pieces = st.selectbox("Pieces", [3, 4, 5, 6], index=1)
         bend = st.number_input("Total Angle", 90); rad = st.number_input("Radius", value=1.5 * float(eval(p_nom.replace("-", "+"))))
         if st.button("🚀 Get Markings"): st.session_state.inputs = {"p_nom": p_nom, "pieces": pieces, "bend": bend, "rad": rad}; st.session_state.step = 3; st.rerun()
    elif st.session_state.tool in ["Miter", "Wye"]:
         p_nom = st.selectbox("Pipe Size", all_sizes, index=8); angle = st.number_input("Angle", 45.0)
         if st.button("🚀 Get Markings"): st.session_state.inputs = {"p_nom": p_nom, "angle": angle}; st.session_state.step = 3; st.rerun()

# ==============================================================================
# STEP 3: MARK (THE ACTION)
# ==============================================================================
elif st.session_state.step == 3:
    if st.button("← Start Over"): reset(); st.rerun()
    st.markdown('<p class="step-header">3. Mark & Cut</p>', unsafe_allow_html=True)
    st.progress(100)
    
    if st.session_state.tool == "Fishmouth":
        d = st.session_state.inputs
        R, r = d['h_od']/2, d['b_od']/2
        if r >= R and d['offset'] == 0: r = R - 0.001
        theta = np.linspace(0, 2*np.pi, 33); alpha = np.radians(d['angle']); x = theta * r
        term = R**2 - (r*np.sin(theta) + d['offset'])**2; term[term<0] = 0
        if d['angle'] == 90: y = np.sqrt(term)
        else: y = (np.sqrt(term)/np.sin(alpha)) + (r*np.cos(theta)/np.tan(alpha))
        y = y - np.min(y)

        # --- TABS FOCUSED ON ACTION ---
        res_tabs = st.tabs(["🔨 How to Mark", "📏 The Numbers", "📷 Check Work"])
        
        with res_tabs[0]:
            st.markdown(f"""
            <div class="instruction-box">
                <b>The 3-Step Marking Process:</b>
            </div>
            """, unsafe_allow_html=True)
            
            st.pyplot(draw_markup_guide())
            
            st.write(f"1. **Draw Base Line:** Use a pipe wrap to draw a straight line around your {d['b_nom']}\" pipe.")
            st.write("2. **Divide:** Fold your pipe wrap to find 16 equal points around the circle.")
            st.write("3. **Measure:** Look at the **'The Numbers'** tab. Measure that distance UP from your base line at each point.")
            st.write("4. **Connect:** Connect your dots to create the cut line.")

        with res_tabs[1]:
            st.write("##### Measure UP from Base Line:")
            
            # Create a more visual table
            df = pd.DataFrame({
                "Line #": [int(i/2)+1 for i in range(0, 33, 2)],
                "Decimal": [round(y[i], 3) for i in range(0, 33, 2)],
                "Fraction (Approx)": [f"{int(y[i])} {int((y[i]%1)*16)}/16" for i in range(0, 33, 2)]
            })
            st.dataframe(df, hide_index=True, use_container_width=True, height=600)

        with res_tabs[2]:
            st.info("Use this to verify your marks before you cut.")
            img_file = st.camera_input("Take Photo of Pipe")
            if img_file:
                image = Image.open(img_file)
                st.write("Adjust Red Line to fit photo:")
                c1, c2 = st.columns(2)
                with c1: scale = st.slider("Zoom", 10, 300, 100); x_shift = st.slider("Move L/R", -500, 500, 0)
                with c2: y_shift = st.slider("Move U/D", -500, 500, 0)
                st.pyplot(plot_overlay_on_image(image, x, y, scale, x_shift, y_shift))

    elif st.session_state.tool == "Lobster":
        d = st.session_state.inputs; p_od = pipe_schedule[d['p_nom']]; num_welds = d['pieces'] - 1
        miter_angle = d['bend'] / (num_welds * 2); long = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] + p_od/2); short = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] - p_od/2)
        st.success(f"Cut {d['pieces']-2} middle pieces."); c1, c2, c3 = st.columns(3)
        c1.metric("Angle", f"{round(miter_angle, 1)}°"); c2.metric("Long", f"{round(long, 3)}\""); c3.metric("Short", f"{round(short, 3)}\"")
    elif st.session_state.tool in ["Miter", "Wye"]:
        d = st.session_state.inputs; cut = np.tan(np.radians(d['angle'])) * pipe_schedule[d['p_nom']]
        st.metric("Cutback", f"{round(cut, 3)}\"")
