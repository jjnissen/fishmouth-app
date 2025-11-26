import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import plotly.graph_objects as go

# --- 1. THE DATABASE ---
pipe_schedule = {
    "Custom Size": 0.0, 
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

# --- STYLING (High Contrast Fixes) ---
st.markdown("""
    <style>
    div.stButton > button:first-child { 
        width: 100%; height: 70px; font-weight: bold; border-radius: 12px; 
        border: 2px solid #e0e0e0; font-size: 18px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    /* Force black text on white box for readability in Dark Mode */
    .instruction-box { 
        background-color: #ffffff; border-left: 6px solid #2196f3; 
        padding: 20px; margin-bottom: 20px; color: #000000 !important; 
        border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .step-header { font-size: 24px; font-weight: 800; color: #0e3c61; margin-bottom: 15px; }
    
    /* Q&A Styling */
    .qa-box {
        background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #eee;
        color: #333 !important;
    }
    .qa-q { font-weight: bold; color: #d32f2f; margin-bottom: 5px; }
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
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.add_patch(patches.Rectangle((0, 0), 6, 3, linewidth=2, edgecolor='#0e3c61', facecolor='white'))
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5)
    ax.text(3, 0.2, "1. DRAW BASE LINE", ha='center', fontsize=10, fontweight='bold')
    x_pos = [1, 2, 3, 4, 5]; heights = [1.0, 2.0, 2.5, 2.0, 1.0]
    for i, x in enumerate(x_pos):
        ax.vlines(x, 0.5, 0.5 + heights[i], color='#2196f3', lw=1)
        ax.plot(x, 0.5 + heights[i], 'ro', markersize=5)
        if i == 2: ax.text(x + 0.1, 1.5, "2. MEASURE UP", color='#2196f3', fontweight='bold')
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

def draw_3d_concept(R, r, offset, angle_deg, theta_pts, y_pts):
    h_len = r * 4; h_theta = np.linspace(0, 2*np.pi, 50); h_x = np.linspace(-h_len/2, h_len/2, 20)
    theta_grid, x_grid = np.meshgrid(h_theta, h_x)
    HY = R * np.cos(theta_grid); HZ = R * np.sin(theta_grid); HX = x_grid
    b_x_surf = r * np.cos(theta_pts); b_y_surf = r * np.sin(theta_pts); b_z_surf = -y_pts
    alpha = np.radians(90 - angle_deg)
    BX = b_x_surf * np.cos(alpha) + b_z_surf * np.sin(alpha)
    BY = b_y_surf + offset
    BZ = -b_x_surf * np.sin(alpha) + b_z_surf * np.cos(alpha)
    BZ = BZ + R + (r if angle_deg < 90 else 0)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=HX, y=HY, z=HZ, opacity=0.3, showscale=False, colorscale='Blues', name='Header'))
    fig.add_trace(go.Scatter3d(x=np.append(BX, BX[0]), y=np.append(BY, BY[0]), z=np.append(BZ, BZ[0]), mode='lines', line=dict(color='yellow', width=6), name='Cut Line'))
    extrude = r * 2; BX_top = BX + extrude * np.sin(alpha); BZ_top = BZ + extrude * np.cos(alpha)
    for i in range(0, len(BX), 2):
        fig.add_trace(go.Scatter3d(x=[BX[i], BX_top[i]], y=[BY[i], BY[i]], z=[BZ[i], BZ_top[i]], mode='lines', line=dict(color='red', width=2), opacity=0.5, showlegend=False))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ==============================================================================
# STEP 1: HOME (The Sales Pitch)
# ==============================================================================
if st.session_state.step == 1:
    st.title("🐟 Fishmouth Pro")
    
    st.info("""
    **Stop Guessing. Start Cutting.**
    
    Grinders are for fixing mistakes. This app is for preventing them. 
    Calculate precise industrial cuts for Pipe (ID) or Tube (OD) in seconds.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🐟 Fishmouth / Tee"): st.session_state.tool = "Fishmouth"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Saddles & Laterals**\nFor welding branches onto headers.")
        
        if st.button("🦞 Lobster Back"): st.session_state.tool = "Lobster"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Segmented Elbows**\nFor making turns when you don't have fittings.")
        
    with c2:
        if st.button("📐 Miter Master"): st.session_state.tool = "Miter"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Simple Angles**\nFor handrails, frames, and roll cages.")
        
        if st.button("Y  True Wye"): st.session_state.tool = "Wye"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Split Flow**\nFor intersecting two pipes at equal angles.")

    st.divider()
    
    # --- Q&A SECTION (Cheeky but Pro) ---
    with st.expander("🤔 Knowledge Base (The 'Why')", expanded=False):
        st.markdown("""
        <div class="qa-box">
            <div class="qa-q">Q: Why do I need 16 lines? Can't I just eyeball it?</div>
            A: You could, if you like gaps. 16 ordinates give you the "Goldilocks" curve—smooth enough to fit tight, but not so many dots that you spend all day marking.
        </div>
        <div class="qa-box">
            <div class="qa-q">Q: What the heck is "Eccentric"?</div>
            A: <b>Concentric</b> means dead center. <b>Eccentric</b> means you slid the pipe to the side (offset). Use this when you need the branch flush with the side of the header (like for a drain).
        </div>
        <div class="qa-box">
            <div class="qa-q">Q: Why can't I select a 6" branch on a 4" header?</div>
            A: Because physics. You can't fit a bucket inside a teacup. The branch must be smaller than or equal to the header.
        </div>
        <div class="qa-box">
            <div class="qa-q">Q: How do I use the results without a printer?</div>
            A: 1. Wrap a strap around the pipe. 2. Fold it in half 4 times (that gives you 16 lines). 3. Mark the pipe. 4. Measure UP from your baseline. Simple.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# STEP 2: MEASURE
# ==============================================================================
elif st.session_state.step == 2:
    if st.button("← Back"): reset(); st.rerun()
    st.markdown('<p class="step-header">2. Configure Cut</p>', unsafe_allow_html=True)
    st.progress(50)
    
    if st.session_state.tool == "Fishmouth":
        c1, c2 = st.columns(2)
        with c1: 
            h_sel = st.selectbox("Header Size", all_sizes, index=12)
            if h_sel == "Custom Size": h_od = st.number_input("Header O.D.", 0.1, 100.0, 4.5); h_nom = f"Custom {h_od}\""
            else: h_od = pipe_schedule[h_sel]; h_nom = h_sel
        with c2: 
            b_sel = st.selectbox("Branch Size", all_sizes, index=10)
            if b_sel == "Custom Size": b_od = st.number_input("Branch O.D.", 0.1, 100.0, 3.5); b_nom = f"Custom {b_od}\""
            else: b_od = pipe_schedule[b_sel]; b_nom = b_sel

        h_type = st.radio("Header Shape", ["Straight Pipe", "Elbow Fitting"], horizontal=True)
        angle = st.number_input("Intersection Angle (°)", 1.0, 90.0, 90.0)

        max_off = max(0.0, (h_od - b_od)/2)
        c_draw, c_input = st.columns([1, 1.5])
        offset = 0.0
        with c_input:
            if max_off <= 0.001: st.slider("Offset (Locked)", 0.0, 1.0, 0.0, disabled=True); st.caption("🔒 Branch ≥ Header")
            else: offset = st.slider("Eccentric Offset", 0.0, max_off, 0.0, step=0.125)
        with c_draw: st.pyplot(draw_concept_visual("ECCENTRIC", h_od, b_od, offset))

        st.divider()
        if st.button("🚀 Calculate Layout"): st.session_state.inputs = {"h_nom": h_nom, "b_nom": b_nom, "h_type": h_type, "angle": angle, "offset": offset, "h_od": h_od, "b_od": b_od}; st.session_state.step = 3; st.rerun()

    elif st.session_state.tool == "Lobster":
         p_nom = st.selectbox("Pipe Size", all_sizes, index=12)
         if p_nom == "Custom Size": p_od = st.number_input("Pipe O.D.", 0.1, 100.0, 4.5)
         else: p_od = pipe_schedule[p_nom]
         pieces = st.selectbox("Pieces", [3, 4, 5, 6], index=1); bend = st.number_input("Total Angle", 90); rad = st.number_input("Radius", value=1.5 * p_od)
         if st.button("🚀 Calculate Layout"): st.session_state.inputs = {"p_nom": p_nom, "pieces": pieces, "bend": bend, "rad": rad, "p_od": p_od}; st.session_state.step = 3; st.rerun()
         
    elif st.session_state.tool in ["Miter", "Wye"]:
         p_nom = st.selectbox("Pipe Size", all_sizes, index=8)
         if p_nom == "Custom Size": p_od = st.number_input("Pipe O.D.", 0.1, 100.0, 4.5)
         else: p_od = pipe_schedule[p_nom]
         angle = st.number_input("Angle", 45.0)
         if st.button("🚀 Calculate Layout"): st.session_state.inputs = {"p_nom": p_nom, "angle": angle, "p_od": p_od}; st.session_state.step = 3; st.rerun()

# ==============================================================================
# STEP 3: RESULTS
# ==============================================================================
elif st.session_state.step == 3:
    if st.button("← Start Over"): reset(); st.rerun()
    st.markdown('<p class="step-header">3. Layout & Mark</p>', unsafe_allow_html=True)
    st.progress(100)
    
    if st.session_state.tool == "Fishmouth":
        d = st.session_state.inputs
        R, r = d['h_od']/2, d['b_od']/2
        if r >= R and d['offset'] == 0: r = R - 0.001
        theta = np.linspace(0, 2*np.pi, 65); alpha = np.radians(d['angle']); x = theta * r
        term = R**2 - (r*np.sin(theta) + d['offset'])**2; term[term<0] = 0
        if d['angle'] == 90: y = np.sqrt(term)
        else: y = (np.sqrt(term)/np.sin(alpha)) + (r*np.cos(theta)/np.tan(alpha))
        y_final = y - np.min(y)

        res_tabs = st.tabs(["🔨 How to Mark", "🌐 3D View", "📏 The Numbers", "📷 Camera Check"])
        
        with res_tabs[0]:
            st.markdown(f"""<div class="instruction-box"><b>Step-by-Step Marking Guide:</b></div>""", unsafe_allow_html=True)
            st.pyplot(draw_markup_guide())
            st.markdown("""
            1.  **Base Line:** Draw a straight ring around your pipe.
            2.  **Divide:** Fold your pipe wrap to split the ring into 16 equal parts.
            3.  **Measure:** Use the numbers in the **'The Numbers'** tab to mark up from the line.
            4.  **Connect:** Connect the dots. Cut on the line.
            """)

        with res_tabs[1]:
            st.write("##### Interactive 3D Model")
            st.caption("Pinch to Zoom. Drag to Rotate.")
            st.plotly_chart(draw_3d_concept(R, r, d['offset'], d['angle'], theta, y), use_container_width=True)

        with res_tabs[2]:
            st.write("##### Measure UP from Base Line:")
            indices = np.linspace(0, 64, 17, dtype=int)
            df = pd.DataFrame({"Line #": range(1, 18), "Decimal": [round(y_final[i], 3) for i in indices], "Fraction (Approx)": [f"{int(y_final[i])} {int((y_final[i]%1)*16)}/16" for i in indices]})
            st.dataframe(df, hide_index=True, use_container_width=True, height=600)

        with res_tabs[3]:
            st.info("Take a photo of your marked pipe to verify the curve.")
            img_file = st.camera_input("Take Photo")
            if img_file:
                image = Image.open(img_file); c1, c2 = st.columns(2)
                with c1: scale = st.slider("Zoom", 10, 300, 100); x_shift = st.slider("Move L/R", -500, 500, 0)
                with c2: y_shift = st.slider("Move U/D", -500, 500, 0)
                st.pyplot(plot_overlay_on_image(image, x, y_final, scale, x_shift, y_shift))

    elif st.session_state.tool == "Lobster":
        d = st.session_state.inputs; p_od = d.get('p_od', pipe_schedule[d['p_nom']]); num_welds = d['pieces'] - 1
        miter_angle = d['bend'] / (num_welds * 2); long = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] + p_od/2); short = 2 * np.tan(np.radians(miter_angle)) * (d['rad'] - p_od/2)
        st.success(f"Cut {d['pieces']-2} middle pieces."); c1, c2, c3 = st.columns(3)
        c1.metric("Angle", f"{round(miter_angle, 1)}°"); c2.metric("Long", f"{round(long, 3)}\""); c3.metric("Short", f"{round(short, 3)}\"")
        
    elif st.session_state.tool in ["Miter", "Wye"]:
        d = st.session_state.inputs; p_od = d.get('p_od', pipe_schedule[d['p_nom']]); cut = np.tan(np.radians(d['angle'])) * p_od
        st.metric("Cutback", f"{round(cut, 3)}\"")
