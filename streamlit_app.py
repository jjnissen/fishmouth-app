import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import io
import requests
from streamlit_lottie import st_lottie
import math

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

# --- ANIMATION LOADER ---
@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_measure = load_lottieurl("https://lottie.host/5a806554-0797-4563-937a-0693296634d4/Q9y3a8g8tC.json")
lottie_print = load_lottieurl("https://lottie.host/9529963a-0202-4662-977b-2993d026df34/z7K1i2Q6Y5.json")

# --- STYLING ---
st.markdown("""
    <style>
    div.stButton > button:first-child { 
        width: 100%; height: 70px; font-weight: bold; border-radius: 12px; 
        border: 2px solid #e0e0e0; font-size: 18px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    .hero-box {
        background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 6px solid #1565c0; 
        margin-bottom: 25px; color: #0d47a1;
    }
    .instruction-box { 
        background-color: #ffffff; border-left: 6px solid #2196f3; 
        padding: 20px; margin-bottom: 20px; color: #000000 !important; 
        border-radius: 8px; border: 1px solid #eee;
    }
    .step-header { font-size: 24px; font-weight: 800; color: #0e3c61; margin-bottom: 15px; }
    .qa-box {
        background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #ddd;
        color: #333 !important;
    }
    .qa-q { font-weight: bold; color: #d32f2f; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- ADVANCED TAPE GENERATOR ---
def generate_advanced_tape(circumference, y_vals, tape_width_inch, mode="RULER"):
    DPI = 203 # Standard Thermal DPI
    img_width = int((circumference + 0.5) * DPI)
    img_height = int(tape_width_inch * DPI)
    
    spacing_px = (circumference * DPI) / 16
    
    # --- MODE A: RULER ---
    if mode == "RULER":
        img = Image.new('RGB', (img_width, img_height), color='white')
        d = ImageDraw.Draw(img)
        for i in range(17): 
            x = int(i * spacing_px)
            d.line([x, 0, x, img_height], fill="black", width=3)
            val_idx = (i * 2) % 32 
            if i == 16: val_idx = 32 
            try: val = y_vals[val_idx]
            except: val = y_vals[0]
            d.text((x + 5, 10), f"#{i+1}", fill="black")
            d.text((x + 5, img_height/2), f"{round(val,3)}\"", fill="red")
        return [img] 

    # --- MODE B: SPLIT STENCIL (MOSAIC) ---
    elif mode == "STENCIL":
        max_y = max(y_vals)
        num_strips = math.ceil(max_y / tape_width_inch)
        if num_strips == 0: num_strips = 1
        
        strips = []
        x_points = np.linspace(0, circumference, len(y_vals))
        
        for s in range(num_strips):
            img = Image.new('RGB', (img_width, img_height), color='white')
            d = ImageDraw.Draw(img)
            y_min = s * tape_width_inch
            y_max = (s + 1) * tape_width_inch
            
            d.line([0, 0, img_width, 0], fill="blue", width=1)
            d.text((10, 5), f"STRIP #{s+1} (Align bottom edge to {y_min}\")", fill="blue")
            
            points = []
            for i in range(len(x_points)):
                phy_x = x_points[i]; phy_y = y_vals[i]
                if phy_y >= (y_min - 0.05) and phy_y <= (y_max + 0.05):
                    px_x = int(phy_x * DPI)
                    rel_y = phy_y - y_min
                    px_y = int(img_height - (rel_y * DPI))
                    points.append((px_x, px_y))
                else:
                    if len(points) > 1: d.line(points, fill="black", width=5)
                    points = []
            if len(points) > 1: d.line(points, fill="black", width=5)
            strips.append(img)
        return strips

# --- VISUAL HELPERS ---
def draw_static_3d_wireframe(R, r, offset, angle_deg):
    fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection='3d')
    h_len = r * 4.5; x = np.linspace(-h_len/2, h_len/2, 15); theta = np.linspace(0, 2*np.pi, 24)
    theta_grid, x_grid = np.meshgrid(theta, x); y_grid = R * np.cos(theta_grid); z_grid = R * np.sin(theta_grid)
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='#90a4ae', alpha=0.4, linewidth=0.5)
    cut_theta = np.linspace(0, 2*np.pi, 100); b_x_surf = r * np.cos(cut_theta); b_y_surf = r * np.sin(cut_theta)
    term_sq = R**2 - (r * np.sin(cut_theta) + offset)**2; term_sq[term_sq < 0] = 0
    alpha_rad = np.radians(angle_deg)
    if angle_deg == 90: cut_depth = np.sqrt(term_sq)
    else: cut_depth = (np.sqrt(term_sq)/np.sin(alpha_rad)) + (r * np.cos(cut_theta)/np.tan(alpha_rad))
    tilt = np.radians(90 - angle_deg); BX = b_x_surf; BY = b_y_surf + offset; BZ = -(cut_depth - np.min(cut_depth))
    BX_rot = BX * np.cos(tilt) + BZ * np.sin(tilt); BZ_rot = -BX * np.sin(tilt) + BZ * np.cos(tilt); BY_rot = BY
    BZ_rot = BZ_rot + R + (r if angle_deg < 90 else 0)
    ax.plot(BX_rot, BY_rot, BZ_rot, color='#ffc107', linewidth=4, zorder=10)
    top_z = BZ_rot + r*3
    for i in range(0, 100, 8): ax.plot([BX_rot[i], BX_rot[i]], [BY_rot[i], BY_rot[i]], [BZ_rot[i], top_z[i]], color='#ef5350', alpha=0.4, linewidth=1)
    ax.plot(BX_rot, BY_rot, top_z, color='#ef5350', alpha=0.4)
    ax.set_axis_off(); ax.view_init(elev=25, azim=-60); ax.set_xlim(-h_len/2, h_len/2); ax.set_ylim(-R*1.5, R*1.5); ax.set_zlim(0, R*5)
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

def draw_concept_visual(mode, h_od, b_od, offset=0):
    fig, ax = plt.subplots(figsize=(4, 3)); ax.set_aspect('equal'); ax.axis('off')
    if mode == "ECCENTRIC":
        ax.add_patch(plt.Circle((0, 0), h_od/2, facecolor='white', edgecolor='#0e3c61', lw=2))
        ax.add_patch(plt.Circle((0, offset), b_od/2, facecolor='#d9eaf7', edgecolor='#2196f3', lw=2, alpha=0.9))
        if offset > 0: ax.annotate('', xy=(0, 0), xytext=(0, offset), arrowprops=dict(arrowstyle='<-', color='red', lw=2))
        ax.set_xlim(-h_od/1.4, h_od/1.4); ax.set_ylim(-h_od/1.4, h_od/1.4)
    return fig

def draw_book_concept(concept_name):
    fig, ax = plt.subplots(figsize=(6, 4)); ax.set_aspect('equal'); ax.axis('off')
    if concept_name == "Page 27: Base Line":
        ax.text(0.5, 0.9, "The Base Line", ha='center', fontweight='bold')
        ax.add_patch(patches.Rectangle((0.2, 0.3), 0.6, 0.4, fill=False, edgecolor='black'))
        ax.plot([0.2, 0.8], [0.4, 0.4], 'k--'); ax.text(0.85, 0.4, "Base Line", fontsize=8)
        for i in np.linspace(0.25, 0.75, 5): ax.arrow(i, 0.4, 0, 0.15, head_width=0.02, color='blue')
    elif concept_name == "Page 40: Eccentric Direction":
        ax.text(0.5, 0.9, "Eccentric Direction", ha='center', fontweight='bold')
        ax.add_patch(plt.Circle((0.2, 0.5), 0.2, fill=False)); ax.add_patch(plt.Circle((0.2, 0.6), 0.1, fill=False))
        ax.text(0.2, 0.2, "Left Hand", ha='center'); ax.add_patch(plt.Circle((0.8, 0.5), 0.2, fill=False))
        ax.add_patch(plt.Circle((0.8, 0.6), 0.1, fill=False)); ax.text(0.8, 0.2, "Right Hand", ha='center')
        ax.annotate("Offset", xy=(0.2, 0.6), xytext=(0.4, 0.6), arrowprops=dict(arrowstyle='->'))
    elif concept_name == "Page 74: Locating Laterals":
        ax.text(0.5, 0.9, "Locating on Header", ha='center', fontweight='bold')
        ax.plot([0, 1], [0.4, 0.4], 'k-'); ax.plot([0, 1], [0.2, 0.2], 'k-'); ax.plot([0, 1], [0.3, 0.3], 'k-.')
        ax.plot([0.4, 0.6], [0.6, 0.4], 'b-'); ax.plot([0.5, 0.7], [0.6, 0.4], 'b-')
        ax.annotate("", xy=(0.6, 0.3), xytext=(0.5, 0.3), arrowprops=dict(arrowstyle='<->', color='red'))
        ax.text(0.55, 0.25, "Measure Dist.", color='red', fontsize=8, ha='center')
    return fig

def plot_overlay_on_image(bg_image, x_vals, y_vals, scale, x_shift, y_shift):
    dpi = 100; height, width = np.array(bg_image).shape[:2]; figsize = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=figsize); ax.imshow(bg_image)
    x_center = np.mean(x_vals); x_scaled = (x_vals - x_center) * scale + (width / 2) + x_shift
    y_scaled = (height / 2) - (y_vals * scale) + y_shift
    ax.plot(x_scaled, y_scaled, color='#ff0000', linewidth=5, alpha=0.8) 
    base_y = (height / 2) + y_shift
    ax.axhline(base_y, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
    ax.axis('off'); return fig

# ==============================================================================
# STEP 1: HOME
# ==============================================================================
if st.session_state.step == 1:
    st.title("🐟 Fishmouth Pro")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if lottie_measure: st_lottie(lottie_measure, height=120, key="intro_anim")
        else: st.image("https://cdn-icons-png.flaticon.com/512/2942/2942076.png", width=100)
            
    with col_b:
        st.markdown("""
        <div class="hero-box">
            <b>Stop Guessing. Start Cutting.</b><br>
            Calculate precise industrial cuts for <b>Pipe (ID)</b> or <b>Tube (OD)</b> in seconds.
        </div>
        """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🐟 Fishmouth / Tee"): st.session_state.tool = "Fishmouth"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Saddles & Laterals**")
        if st.button("🦞 Lobster Back"): st.session_state.tool = "Lobster"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Segmented Elbows**")
    with c2:
        if st.button("📐 Miter Master"): st.session_state.tool = "Miter"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **Simple Angles**")
        if st.button("📖 Book Guide"): st.session_state.tool = "Book"; st.session_state.step = 2; st.rerun()
        st.caption("✅ **The Manual**")
    st.divider()
    with st.expander("🤔 Knowledge Base (The 'Why')", expanded=False):
        st.markdown("""
        <div class="qa-box"><div class="qa-q">Q: Why 16 lines?</div>A: It's the "Goldilocks" curve—smooth enough to fit tight, not too many to mark.</div>
        <div class="qa-box"><div class="qa-q">Q: What is "Eccentric"?</div>A: Offset to the side (not centered).</div>
        <div class="qa-box"><div class="qa-q">Q: How do I use the "Smart Tape"?</div>A: Use any cheap Bluetooth thermal printer. If you have a 4-inch printer (like a Phomemo M04S), it prints one perfect stencil. If you have a small printer, it prints stacking strips.</div>
        """, unsafe_allow_html=True)

# ==============================================================================
# STEP 2: MEASURE
# ==============================================================================
elif st.session_state.step == 2:
    if st.button("← Back"): reset(); st.rerun()
    
    if st.session_state.tool == "Book":
        st.markdown('<p class="step-header">📖 Reference Gallery</p>', unsafe_allow_html=True)
        page = st.selectbox("Select Concept:", ["Page 27: Base Line", "Page 40: Eccentric Direction", "Page 74: Locating Laterals"])
        st.pyplot(draw_book_concept(page))
        if page == "Page 27: Base Line": st.write("**Rule:** Always measure UP from the Base Line.")
        elif page == "Page 40: Eccentric Direction": st.write("**Rule:** Align lowest point with offset side.")
        elif page == "Page 74: Locating Laterals": st.write("**Rule:** Use the center line of the header.")

    elif st.session_state.tool == "Fishmouth":
        st.markdown('<p class="step-header">2. Configure Cut</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: 
            h_sel = st.selectbox("Header", all_sizes, index=12)
            if h_sel == "Custom Size": h_od = st.number_input("Header O.D.", 0.1, 100.0, 4.5); h_nom = f"Custom {h_od}\""
            else: h_od = pipe_schedule[h_sel]; h_nom = h_sel
        with c2: 
            b_sel = st.selectbox("Branch", all_sizes, index=10)
            if b_sel == "Custom Size": b_od = st.number_input("Branch O.D.", 0.1, 100.0, 3.5); b_nom = f"Custom {b_od}\""
            else: b_od = pipe_schedule[b_sel]; b_nom = b_sel

        h_type = st.radio("Header Shape", ["Straight Pipe", "Elbow Fitting"], horizontal=True)
        angle = st.number_input("Angle (°)", 1.0, 90.0, 90.0)
        max_off = max(0.0, (h_od - b_od)/2)
        c_draw, c_input = st.columns([1, 1.5])
        offset = 0.0
        with c_input:
            if max_off <= 0.001: st.slider("Offset (Locked)", 0.0, 1.0, 0.0, disabled=True); st.caption("🔒 Branch ≥ Header")
            else: offset = st.slider("Eccentric Offset", 0.0, max_off, 0.0, step=0.125)
        with c_draw: st.pyplot(draw_concept_visual("ECCENTRIC", h_od, b_od, offset))
