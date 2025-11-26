import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

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

st.markdown("""
    <style>
    div.stButton > button:first-child { width: 100%; height: 65px; font-weight: bold; border-radius: 10px; border: 1px solid #d1d9e6; font-size: 18px; }
    .instruction-box { background-color: #fff; border-left: 5px solid #2196f3; padding: 15px; margin-bottom: 20px; color: #333; border: 1px solid #eee; }
    .step-header { font-size: 24px; font-weight: bold; color: #0e3c61; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- STATIC 3D WIREFRAME GENERATOR (Works on ALL Phones) ---
def draw_static_3d_wireframe(R, r, offset, angle_deg):
    """Draws a 'Book Style' 3D Wireframe using standard Matplotlib"""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Header Mesh (The Blue Pipe)
    h_len = r * 4.5
    x = np.linspace(-h_len/2, h_len/2, 15)
    theta = np.linspace(0, 2*np.pi, 24) # 24 segments = "Triangles" look
    theta_grid, x_grid = np.meshgrid(theta, x)
    y_grid = R * np.cos(theta_grid)
    z_grid = R * np.sin(theta_grid)
    # Draw grey wireframe
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='#cfd8dc', alpha=0.5, linewidth=0.5)

    # 2. Calculate Intersection Curve
    cut_theta = np.linspace(0, 2*np.pi, 100)
    b_x_surf = r * np.cos(cut_theta)
    b_y_surf = r * np.sin(cut_theta)
    
    term_sq = R**2 - (r * np.sin(cut_theta) + offset)**2
    term_sq[term_sq < 0] = 0
    alpha_rad = np.radians(angle_deg)
    
    if angle_deg == 90: cut_depth = np.sqrt(term_sq)
    else: cut_depth = (np.sqrt(term_sq)/np.sin(alpha_rad)) + (r * np.cos(cut_theta)/np.tan(alpha_rad))
    
    # Transform for 3D Plotting
    tilt = np.radians(90 - angle_deg)
    BX = b_x_surf
    BY = b_y_surf + offset
    BZ = -(cut_depth - np.min(cut_depth))
    
    # Rotate Branch
    BX_rot = BX * np.cos(tilt) + BZ * np.sin(tilt)
    BZ_rot = -BX * np.sin(tilt) + BZ * np.cos(tilt)
    BY_rot = BY
    
    # Shift Up
    BZ_rot = BZ_rot + R + (r if angle_deg < 90 else 0) # Sit on top

    # 3. Draw The Cut Line (Bold Yellow)
    ax.plot(BX_rot, BY_rot, BZ_rot, color='#ffc107', linewidth=4, zorder=10)
    
    # 4. Draw Branch Wireframe (Red)
    top_z = BZ_rot + r*3
    # Draw vertical stringers
    for i in range(0, 100, 8):
        ax.plot([BX_rot[i], BX_rot[i]], [BY_rot[i], BY_rot[i]], [BZ_rot[i], top_z[i]], color='#ef5350', alpha=0.4, linewidth=1)
    # Draw top ring
    ax.plot(BX_rot, BY_rot, top_z, color='#ef5350', alpha=0.4)

    # Setup "Blueprint" Look
    ax.set_axis_off()
    ax.view_init(elev=25, azim=-60) # Classic Isometric View
    ax.set_xlim(-h_len/2, h_len/2); ax.set_ylim(-R*1.5, R*1.5); ax.set_zlim(0, R*5)
    return fig

def draw_miter_3d(p_od, angle):
    """Draws a simple 3D Miter Cut"""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tube Body
    length = p_od * 3
    z = np.linspace(0, length, 10)
    theta = np.linspace(0, 2*np.pi, 24)
    z_grid, theta_grid = np.meshgrid(z, theta)
    x_grid = (p_od/2) * np.cos(theta_grid)
    y_grid = (p_od/2) * np.sin(theta_grid)
    
    # Miter Cut Plane logic
    # We just draw the tube and then a cut plane for visual
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='#cfd8dc', alpha=0.5, linewidth=0.5)
    
    # Draw the cut face ellipse
    cut_h = np.tan(np.radians(angle)) * p_od
    # Ellipse points
    t = np.linspace(0, 2*np.pi, 100)
    ex = (p_od/2) * np.cos(t)
    ey = (p_od/2) * np.sin(t)
    # Slope equation Z = y * tan(angle) + offset
    ez = ey * np.tan(np.radians(angle)) + length - cut_h
    
    ax.plot(ex, ey, ez, color='#d32f2f', linewidth=3)
    
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    return fig

# --- BOOK DIAGRAM REPLICATOR ---
def draw_book_concept(concept_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect('equal'); ax.axis('off')
    
    if concept_name == "Page 40: Eccentric Direction":
        # Replicating Page 40
        ax.text(0.5, 0.9, "Eccentric Lateral Direction", ha='center', fontweight='bold')
        # Left Hand Diagram
        c1 = plt.Circle((0.2, 0.5), 0.2, fill=False, edgecolor='black')
        c2 = plt.Circle((0.2, 0.6), 0.1, fill=False, edgecolor='black')
        ax.add_patch(c1); ax.add_patch(c2)
        ax.text(0.2, 0.2, "Left Hand", ha='center')
        # Right Hand Diagram
        c3 = plt.Circle((0.8, 0.5), 0.2, fill=False, edgecolor='black')
        c4 = plt.Circle((0.8, 0.6), 0.1, fill=False, edgecolor='black')
        ax.add_patch(c3); ax.add_patch(c4)
        ax.text(0.8, 0.2, "Right Hand", ha='center')
        # Arrows
        ax.annotate("Offset", xy=(0.2, 0.6), xytext=(0.4, 0.6), arrowprops=dict(arrowstyle='->'))
        
    elif concept_name == "Page 74: Locating Laterals":
        # Replicating Page 74 Header Centerline
        ax.text(0.5, 0.9, "Locating on Header Pipe", ha='center', fontweight='bold')
        # Header Pipe
        ax.plot([0, 1], [0.4, 0.4], 'k-') # Top
        ax.plot([0, 1], [0.2, 0.2], 'k-') # Bottom
        ax.plot([0, 1], [0.3, 0.3], 'k-.') # Centerline
        # Branch
        ax.plot([0.4, 0.6], [0.6, 0.4], 'b-')
        ax.plot([0.5, 0.7], [0.6, 0.4], 'b-')
        # Dimension
        ax.annotate("", xy=(0.6, 0.3), xytext=(0.5, 0.3), arrowprops=dict(arrowstyle='<->', color='red'))
        ax.text(0.55, 0.25, "Measure this distance", color='red', fontsize=8, ha='center')
        
    elif concept_name == "Page 27: Base Line":
        # Replicating Page 27 Layout
        ax.text(0.5, 0.9, "The Base Line", ha='center', fontweight='bold')
        rect = patches.Rectangle((0.2, 0.3), 0.6, 0.4, fill=False, edgecolor='black')
        ax.add_patch(rect)
        ax.plot([0.2, 0.8], [0.4, 0.4], 'k--') # Base Line
        ax.text(0.85, 0.4, "Base Line", fontsize=8)
        # Arrows up
        for i in np.linspace(0.25, 0.75, 5):
            ax.arrow(i, 0.4, 0, 0.15, head_width=0.02, color='blue')
            
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
# STEP 1: HOME
# ==============================================================================
if st.session_state.step == 1:
    st.title("🐟 Fishmouth Pro")
    st.markdown("""<div class="instruction-box"><b>Universal Pipe & Tube Calculator</b><br>For Plumbers, Welders, and Fabricators.</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🐟 Tee / Lateral"): st.session_state.tool = "Fishmouth"; st.session_state.step = 2; st.rerun()
        if st.button("🦞 Segmented Elbow"): st.session_state.tool = "Lobster"; st.session_state.step = 2; st.rerun()
    with c2:
        if st.button("📐 Simple Miter"): st.session_state.tool = "Miter"; st.session_state.step = 2; st.rerun()
        if st.button("📖 Book Guide"): st.session_state.tool = "Book"; st.session_state.step = 2; st.rerun()

# ==============================================================================
# STEP 2: MEASURE (OR BOOK)
# ==============================================================================
elif st.session_state.step == 2:
    if st.button("← Back"): reset(); st.rerun()
    
    if st.session_state.tool == "Book":
        st.markdown('<p class="step-header">📖 Book Reference Gallery</p>', unsafe_allow_html=True)
        st.info("Concepts and Diagrams from the Fishmouth Manual.")
        
        page = st.selectbox("Select Concept:", ["Page 27: Base Line", "Page 40: Eccentric Direction", "Page 74: Locating Laterals"])
        st.pyplot(draw_book_concept(page))
        
        if page == "Page 27: Base Line":
            st.write("**The Rule:** Always measure UP from the Base Line. Never measure down from the pipe end.")
        elif page == "Page 40: Eccentric Direction":
            st.write("**The Rule:** If the cut is Eccentric (Offset), align the lowest point of the curve with the side of the header you are offsetting towards.")
        elif page == "Page 74: Locating Laterals":
            st.write("**The Rule:** Use the center line of the header. The 'Fishmouth' tool calculates the cut, but you must calculate where to place it on the header (See Layout Tab in Calculator).")

    elif st.session_state.tool == "Fishmouth":
        st.markdown('<p class="step-header">2. Measure & Input</p>', unsafe_allow_html=True)
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
        
        # STATIC 3D PREVIEW (Safe for mobile)
        with c_draw:
            st.caption("3D Wireframe Preview:")
            st.pyplot(draw_static_3d_wireframe(h_od/2, b_od/2, offset, angle))

        st.divider()
        if st.button("🚀 Get Markings"): st.session_state.inputs = {"h_nom": h_nom, "b_nom": b_nom, "h_type": h_type, "angle": angle, "offset": offset, "h_od": h_od, "b_od": b_od}; st.session_state.step = 3; st.rerun()

    elif st.session_state.tool == "Lobster":
         p_nom = st.selectbox("Pipe Size", all_sizes, index=12)
         if p_nom == "Custom Size": p_od = st.number_input("Pipe O.D.", 0.1, 100.0, 4.5)
         else: p_od = pipe_schedule[p_nom]
         pieces = st.selectbox("Pieces", [3, 4, 5, 6], index=1); bend = st.number_input("Total Angle", 90); rad = st.number_input("Radius", value=1.5 * p_od)
         if st.button("🚀 Get Markings"): st.session_state.inputs = {"p_nom": p_nom, "pieces": pieces, "bend": bend, "rad": rad, "p_od": p_od}; st.session_state.step = 3; st.rerun()
         
    elif st.session_state.tool in ["Miter", "Wye"]:
         p_nom = st.selectbox("Pipe Size", all_sizes, index=8)
         if p_nom == "Custom Size": p_od = st.number_input("Pipe O.D.", 0.1, 100.0, 4.5)
         else: p_od = pipe_schedule[p_nom]
         angle = st.number_input("Angle", 45.0)
         if st.button("🚀 Get Markings"): st.session_state.inputs = {"p_nom": p_nom, "angle": angle, "p_od": p_od}; st.session_state.step = 3; st.rerun()

# ==============================================================================
# STEP 3: RESULTS
# ==============================================================================
elif st.session_state.step == 3:
    if st.button("← Start Over"): reset(); st.rerun()
    st.markdown('<p class="step-header">3. Mark & Cut</p>', unsafe_allow_html=True)
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

        res_tabs = st.tabs(["🔨 How to Mark", "🌐 3D View", "📏 The Numbers", "📷 Check Work"])
        
        with res_tabs[0]:
            st.markdown(f"""<div class="instruction-box"><b>Marking Guide for {d['b_nom']} Pipe:</b></div>""", unsafe_allow_html=True)
            st.pyplot(draw_markup_guide())
            st.write("1. **Base Line:** Draw a ring around the pipe.")
            st.write("2. **Divide:** Mark 16 points.")
            st.write("3. **Measure:** Use the numbers in the next tab.")

        with res_tabs[1]:
            st.write("##### 3D Wireframe (Static)")
            st.pyplot(draw_static_3d_wireframe(R, r, d['offset'], d['angle']))

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
        st.pyplot(draw_miter_3d(p_od, d['angle']))
