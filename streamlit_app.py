import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. THE DATABASE (From Pages 6-9 of your Book) ---
# Format: Nominal Size: Outside Diameter (OD)
pipe_schedule = {
    "1/8": 0.405, "1/4": 0.540, "3/8": 0.675, "1/2": 0.840,
    "3/4": 1.050, "1": 1.315, "1-1/4": 1.660, "1-1/2": 1.900,
    "2": 2.375, "2-1/2": 2.875, "3": 3.500, "3-1/2": 4.000,
    "4": 4.500, "5": 5.563, "6": 6.625, "8": 8.625,
    "10": 10.750, "12": 12.750, "14": 14.000, "16": 16.000,
    "18": 18.000, "20": 20.000, "24": 24.000
}

# --- 2. THE APP INTERFACE ---
st.set_page_config(page_title="Fishmouth Pro", page_icon="🐟")

# Header with Logo styling
st.title("🐟 Fishmouth Pro")
st.caption("Computer Calculated Pipe Template Measures")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Header Pipe")
    header_nom = st.selectbox("Header Size", list(pipe_schedule.keys()), index=12) # Default to 4"
    header_od = pipe_schedule[header_nom]
    st.metric("Header O.D.", f"{header_od}\"")

with col2:
    st.subheader("Branch Pipe")
    branch_nom = st.selectbox("Branch Size", list(pipe_schedule.keys()), index=10) # Default to 3"
    branch_od = pipe_schedule[branch_nom]
    st.metric("Branch O.D.", f"{branch_od}\"")

st.divider()

col3, col4 = st.columns(2)
with col3:
    angle = st.number_input("Angle (Degrees)", min_value=1.0, max_value=90.0, value=90.0)
with col4:
    precision = st.radio("Resolution (Ordinates)", [16, 32], horizontal=True)

# --- 3. THE CALCULATION ENGINE (The Math) ---

def calculate_fishmouth(R_header, R_branch, angle_deg, num_points):
    # Convert inputs to radius and radians
    R = R_header / 2
    r = R_branch / 2
    alpha = np.radians(angle_deg)
    
    # THE LIE RULE (Page 21 Logic)
    # If pipes are same size, we treat the branch as slightly smaller 
    # or use OD logic to ensure it straddles the header.
    # In this math model, if r >= R, we cap r at R - 0.001 to prevent math errors
    if r >= R:
        st.warning(f"⚠️ Full Size Cut Detected. Applying 'Lie Rule' (Using OD for fit).")
        r = R - 0.001

    # Generate angles around the branch pipe (0 to 360)
    theta = np.linspace(0, 2 * np.pi, num_points + 1)
    
    # FISHMOUTH FORMULA (Cylindrical Intersection)
    # This calculates the curve length needed to wrap around the branch
    
    # 1. Unwrapped X coordinate (Circumference)
    x_vals = theta * r 
    
    # 2. Y coordinate (The Cut Depth)
    # Simplified general intersection formula for Lateral/Tee
    term1 = np.sqrt(R**2 - (r * np.sin(theta))**2)
    
    if angle_deg == 90:
        y_vals = term1
    else:
        # Complex Lateral Math
        # y = (sqrt(R^2 - (r sin t)^2) / sin a) + (r cos t / tan a)
        y_vals = (term1 / np.sin(alpha)) + (r * np.cos(theta) / np.tan(alpha))

    # Normalize y to start at 0 for the template bottom
    y_vals = y_vals - np.min(y_vals)
    
    return x_vals, y_vals, theta

# Run Calculation
if branch_od > header_od:
    st.error("Error: Branch cannot be larger than Header.")
else:
    x, y, theta = calculate_fishmouth(header_od, branch_od, angle, precision)

    # --- 4. THE OUTPUT (Visual & Data) ---
    
    st.subheader("Template Layout")
    
    # Create the Plot (Visualizing Page 37)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="black", linewidth=2)
    ax.fill_between(x, y, color="skyblue", alpha=0.3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("Circumference (inches)")
    ax.set_ylabel("Ordinate Measure (inches)")
    ax.set_title(f"Cut Template for {branch_nom}\" Branch on {header_nom}\" Header")
    
    # Draw Ordinate Lines
    for i in range(len(x)):
        ax.vlines(x[i], 0, y[i], color="gray", linestyle=':', linewidth=0.5)
        # Label specific points to match the book
        if i < len(x)-1: # Don't label the last duplicate point
             ax.text(x[i], -0.2, f"{i+1}", ha='center', fontsize=8)

    st.pyplot(fig)

    # Display the "Cheat Sheet" Numbers
    st.subheader("Ordinate Measures (The Numbers)")
    st.info("Mark your pipe into equal sections and measure UP from the Base Line.")
    
    # Format data for table
    df = pd.DataFrame({
        "Ordinate Line #": range(1, len(y)),
        "Measure (Inches)": np.round(y[:-1], 3)
    })
    
    # Show as a clean table (interactive)
    st.dataframe(df, use_container_width=True)

    # Print Button Simulation
    st.button("📄 Generate PDF Template (Coming Soon)")
