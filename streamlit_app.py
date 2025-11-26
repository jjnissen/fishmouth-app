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

# --- MOSAIC TAPE GENERATOR ---
def generate_mosaic_strips(R, r, offset, angle, tape_width_inch):
    DPI = 203 # Standard Thermal DPI
    
    # 1. Generate High-Res Curve Data
    theta = np.linspace(0, 2*np.pi, 1000)
    term_sq = R**2 - (r * np.sin(theta) + offset)**2
    term_sq[term_sq < 0] = 0
    alpha_rad = np.radians(angle)
    
    if angle == 90: y_raw = np.sqrt(term_sq)
    else: y_raw = (np.sqrt(term_sq)/np.sin(alpha_rad)) + (r * np.cos(theta)/np.tan(alpha_rad))
    
    y_vals = y_raw - np.min(y_raw)
    circumference = 2 * np.pi * r
    x_vals = np.linspace(0, circumference, 1000)
    
    # 2. Calculate Strips
    max_y = np.max(y_vals)
    num_strips = math.ceil(max_y / tape_width_inch)
    if num_strips == 0: num_strips = 1
    
    strips = []
    img_width = int((circumference + 0.1) * DPI)
    img_height = int(tape_width_inch * DPI)
    
    for s in range(num_strips):
        img = Image.new('RGB', (img_width, img_height), color='white')
        d = ImageDraw.Draw(img)
        
        y_min_phy = s * tape_width_inch
        y_max_phy = (s + 1) * tape_width_inch
        
        # Guide Lines
        label = f"STRIP #{s+1} (Bottom)" if s == 0 else f"STRIP #{s+1} (Stack on #{s})"
        d.text((10, 5), label, fill="blue")
        d.line([0, 0, img_width, 0], fill="blue", width=1) # Top edge for alignment
        
        # Draw Curve Segment
        points_to_draw = []
        for i in range(len(x_vals)):
            phy_x = x_vals[i]; phy_y = y_vals[i]
            if phy_y >= (y_min_phy - 0.05) and phy_y <= (y_max_phy + 0.05):
                px_x = int(phy_x * DPI)
                # Invert Y for image coords
                rel_y = phy_y - y_min_phy
                px_y = int(img_height - (rel_y * DPI))
                points_to_draw.append((px_x, px_y))
            else:
                if len(points_to_draw) > 1: d.line(points_to_draw, fill="black", width=5)
                points_to_draw = []
        if len(points_to_draw) > 1: d.line(points_to_draw, fill="black", width=5)
        strips.append(img)
        
    return strips, num_strips, max_y

# --- VISUAL HELPERS ---
def draw_mosaic_guide():
    fig, ax = plt.subplots(figsize=(6, 3))
    rect = patches.Rectangle((0, 0), 6, 3, linewidth=2, edgecolor='#0e3c61', facecolor='#e3f2fd')
    ax.add_patch(rect)
    ax.text(3, 1.5, "PIPE", ha='center', color='#0e3c61', alpha=0.2, fontweight='bold', fontsize=20)
    # Strip 1
    rect_1 = patches.Rectangle((0, 0.5), 6, 0.5, linewidth=1, edgecolor='blue', facecolor='#fff9c4', alpha=0.9)
    ax.add_patch(rect_1)
    ax.text(3, 0.75, "STRIP #1 (Base)", ha='center', fontsize=8)
    ax.plot([1, 2, 3], [0.5, 0.8, 1.0], color='black', linewidth=2)
    # Strip 2
    rect_2 = patches.Rectangle((0, 1.0), 6, 0.5, linewidth=1, edgecolor='blue', facecolor='#fff9c4', alpha=0.9)
    ax.add_patch(rect_2)
    ax.text(3, 1.25, "STRIP #2 (Stack on Top)", ha='center', fontsize=8)
    ax.plot([3, 4, 5], [1.0, 1.2, 1.0], color='black', linewidth=2)
    
    ax.text(3, 0.2, "Print Strips → Stack Them → Cut on Line", ha='center', fontsize=10, fontweight='bold', color='#0e3c61')
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(0, 3.0); ax.axis('off')
    return fig

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
        st.markdown("""<div class="hero-box"><b>Stop Guessing. Start Cutting.</b><br>Calculate precise industrial cuts for <b>Pipe (ID)</b> or <b>Tube (OD)</b> in seconds.</div>""", unsafe_allow_html=True)
    
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
    with st.expander("🤔 Knowledge Base", expanded=False):
        st.markdown("""<div class="qa-box"><div class="qa-q">Q: Why 16 lines?</div>A: It's the "Goldilocks" curve—smooth enough to fit tight, not too many to mark.</div><div class="qa-box"><div class="qa-q">Q: How do I use the "Smart Tape"?</div>A: Use any cheap Bluetooth thermal printer. If the cut is deep, check <b>"Force Split Stencil"</b> to print stackable strips.</div>""", unsafe_allow_html=True)

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
        st.divider()
        if st.button("🚀 Calculate"): st.session_state.inputs = {"h_nom": h_nom, "b_nom": b_nom, "h_type": h_type, "angle": angle, "offset": offset, "h_od": h_od, "b_od": b_od}; st.session_state.step = 3; st.rerun()

    elif st.session_state.tool == "Lobster":
         p_nom = st.selectbox("Pipe Size", all_sizes, index=12); 
         if p_nom=="Custom Size": p_od = st.number_input("OD", 0.1, 100.0, 4.5)
         else: p_od = pipe_schedule[p_nom]
         pieces = st.selectbox("Pieces", [3, 4, 5, 6], index=1); bend = st.number_input("Angle", 90); rad = st.number_input("Radius", value=1.5 * p_od)
         if st.button("Calculate"): st.session_state.inputs = {"p_nom": p_nom, "pieces": pieces, "bend": bend, "rad": rad, "p_od": p_od}; st.session_state.step = 3; st.rerun()
    elif st.session_state.tool in ["Miter", "Wye"]:
         p_nom = st.selectbox("Pipe Size", all_sizes, index=8); 
         if p_nom=="Custom Size": p_od = st.number_input("OD", 0.1, 100.0, 4.5)
         else: p_od = pipe_schedule[p_nom]
         angle = st.number_input("Angle", 45.0)
         if st.button("Calculate"): st.session_state.inputs = {"p_nom": p_nom, "angle": angle, "p_od": p_od}; st.session_state.step = 3; st.rerun()

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

        res_tabs = st.tabs(["🖨️ Smart Tape", "🔨 Mark", "🌐 3D", "📏 Data", "📷 Camera"])
        
        with res_tabs[0]:
            c_anim, c_text = st.columns([1, 3])
            with c_anim: 
                if lottie_print: st_lottie(lottie_print, height=80, key="print_anim")
            with c_text: st.markdown(f"""<div class="instruction-box"><b>The Mosaic Stencil:</b><br>If curve is too deep, use the 'Split Stencil' box to create stackable strips.</div>""", unsafe_allow_html=True)
            
            st.pyplot(draw_mosaic_guide()) # GUIDE
            
            # --- MOSAIC LOGIC ---
            tape_width = st.select_slider("Printer Tape Width:", options=[0.5, 1.0, 1.5, 2.0, 4.0], value=0.5)
            force_split = st.checkbox("Force Split Stencil (Multi-Strip)", help="Split the curve into multiple stackable strips")
            
            strips, num_strips, max_h = generate_mosaic_strips(R, r, d['offset'], d['angle'], tape_width)
            
            # Display strips
            st.write(f"**Cut Height:** {round(max_h, 2)}\" | **Strips Required:** {num_strips}")
            
            for i, img in enumerate(strips):
                st.write(f"**Strip #{i+1}**")
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                st.image(img, caption=f"Strip {i+1}")
                st.download_button(f"📥 Download Strip {i+1}", buf.getvalue(), file_name=f"strip_{i+1}.png", mime="image/png")

        with res_tabs[1]:
            st.markdown(f"""<div class="instruction-box"><b>Manual Marking Guide:</b></div>""", unsafe_allow_html=True)
            st.pyplot(draw_markup_guide())
            st.write("1. **Base Line:** Draw a straight ring around your pipe.")
            st.write("2. **Divide:** Fold your pipe wrap to split the ring into 16 equal parts.")
            st.write("3. **Measure:** Measure UP from the line using the 'Data' tab numbers.")

        with res_tabs[2]:
            st.write("##### 3D Visualization")
            st.pyplot(draw_static_3d_wireframe(R, r, d['offset'], d['angle']))

        with res_tabs[3]:
            st.write("##### Measure UP from Base Line:")
            indices = np.linspace(0, 64, 17, dtype=int)
            df = pd.DataFrame({"Line #": range(1, 18), "Decimal": [round(y_final[i], 3) for i in indices], "Fraction (Approx)": [f"{int(y_final[i])} {int((y_final[i]%1)*16)}/16" for i in indices]})
            st.dataframe(df, hide_index=True, use_container_width=True, height=600)

        with res_tabs[4]:
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
