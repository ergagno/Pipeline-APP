import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import numpy as np

# Function to create the map
def create_map(df):
    m = folium.Map(
        location=[df['Latitude'].mean(), df['Longitude'].mean()],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    girthweld_df = df[df['Feature Type'] == 'GirthWeld']
    for _, row in girthweld_df.iterrows():
        folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius=10,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6,
            popup=f"GirthWeld {row['Feature ID']}"
        ).add_to(m)
    
    metal_loss_df = df[df['Feature Type'] == 'Metal Loss']
    for _, row in metal_loss_df.iterrows():
        rpr = row['RPR Eff. Area 1.25 MAOP']
        color = 'red' if rpr < 1.5 else 'orange' if rpr < 1.75 else 'yellow'
        folium.Circle(
            location=[row['Latitude'], row['Longitude']],
            radius=15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Metal Loss {row['Feature ID']}<br>RPR: {rpr:.2f}"
        ).add_to(m)
    
    return m

# Modified B31G Burst Pressure Calculation
def calculate_burst_pressure(d, L, t, D, SMYS, MAOP):
    if any(pd.isna(x) or x <= 0 for x in [d, L, t, D, SMYS, MAOP]):
        return np.nan
    
    F = SMYS + 10000
    A = 0.893 * (L / np.sqrt(D * t))
    M = np.sqrt(1 + 0.6275 * A**2 - 0.003375 * A**4) if A <= 4.0 else 0.032 * A + 3.3
    d_t_ratio = d / t
    P_f = (2 * F * t / D) * ((1 - 0.85 * d_t_ratio) / (1 - 0.85 * d_t_ratio / M))
    return max(P_f, 0)

# Risk Assessment Function
def calculate_risk(row, df):
    rpr = row['RPR Eff. Area 1.25 MAOP']
    depth_ratio = row['Peak Depth'] / row['Nominal Wall (in) Thickness']
    growth_rate = 0.01
    time_to_failure = (row['Nominal Wall (in) Thickness'] * 0.8 - row['Peak Depth']) / growth_rate if row['Peak Depth'] < row['Nominal Wall (in) Thickness'] * 0.8 else 0
    likelihood = 5 if rpr < 0.2 or depth_ratio > 0.8 else 4 if rpr < 1.5 else 3 if rpr < 1.75 else 1

    lat, lon = row['Latitude'], row['Longitude']
    nearby_features = df[
        (df['Latitude'].between(lat - 0.0001, lat + 0.0001)) & 
        (df['Longitude'].between(lon - 0.0001, lon + 0.0001)) &
        (df['Feature Type'].isin(['Support', 'Valve', 'Bend', 'Branch Connection']))
    ]
    consequence = 5 if len(nearby_features) > 2 else 3 if len(nearby_features) > 0 else 1

    risk_score = likelihood * consequence
    
    if rpr < 0.2 or (not pd.isna(time_to_failure) and time_to_failure <= 1):  # Immediate if RPR < 0.2 or failure within 1 year
        priority = "Immediate"
    elif risk_score >= 15:
        priority = "High"
    elif risk_score >= 5:
        priority = "Medium"
    else:
        priority = "Low"
    
    return likelihood, consequence, risk_score, priority

# Main app
def main():
    st.set_page_config(
        page_title="Pipeline Inspection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main {background-color: #f5f6f5; padding: 20px;}
        .stButton>button {background-color: #0066cc; color: white; border-radius: 5px;}
        .stDataFrame {border: 1px solid #ddd; border-radius: 5px;}
        </style>
    """, unsafe_allow_html=True)

    st.title("Pipeline Inspection Dashboard")
    st.markdown("Interactive visualization of pipeline inspection data")

    with st.sidebar:
        st.header("Controls")
        feature_types = st.multiselect(
            "Filter Feature Types",
            options=['GirthWeld', 'Metal Loss'],
            default=['GirthWeld', 'Metal Loss']
        )

    try:
        df = pd.read_excel("20-inch EL100 Pipetally.xlsx", engine='openpyxl')
        filtered_df = df[df['Feature Type'].isin(feature_types)]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Pipeline Map")
            m = create_map(filtered_df)
            folium_static(m, width=700, height=500)
            
            st.markdown("""
                ### Legend
                - 🟢 Green: Girth Welds
                - 🔴 Red: Metal Loss (RPR < 1.5)
                - 🟠 Orange: Metal Loss (1.5 ≤ RPR < 1.75)
                - 🟡 Yellow: Metal Loss (RPR ≥ 1.75)
            """)
            
            # Burst Pressure and Risk Analysis for ALL Metal Loss features
            st.subheader("Burst Pressure & Risk Analysis (All Metal Loss Features)")
            metal_loss_df = filtered_df[filtered_df['Feature Type'] == 'Metal Loss']
            
            if not metal_loss_df.empty:
                analysis_data = []
                for _, row in metal_loss_df.iterrows():
                    P_f = calculate_burst_pressure(
                        d=row['Peak Depth'],
                        L=row['Length'],
                        t=row['Nominal Wall (in) Thickness'],
                        D=row['NPS'],
                        SMYS=row['SMYS'],
                        MAOP=row['MOP/MAOP']
                    )
                    likelihood, consequence, risk_score, priority = calculate_risk(row, filtered_df)
                    
                    analysis_data.append({
                        'Feature ID': row['Feature ID'],
                        'Latitude': row['Latitude'],
                        'Longitude': row['Longitude'],
                        'Depth (in)': row['Peak Depth'],
                        'Length (in)': row['Length'],
                        'Burst Pressure (psi)': P_f,
                        'MAOP (psi)': row['MOP/MAOP'],
                        'Safe?': 'Yes' if P_f > row['MOP/MAOP'] else 'No',
                        'Likelihood': likelihood,
                        'Consequence': consequence,
                        'Risk Score': risk_score,
                        'Priority': priority
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                # Sort table: "No" (not safe) at the top
                analysis_df = analysis_df.sort_values(by='Safe?', ascending=True)
                
                def apply_safe_color(row):
                    colors = [''] * len(row)
                    if row['Safe?'] == 'No':
                        colors[analysis_df.columns.get_loc('Safe?')] = 'color: red'
                    elif row['Safe?'] == 'Yes' and row['Priority'] == 'Immediate':
                        colors[analysis_df.columns.get_loc('Safe?')] = 'color: red'
                    elif row['Safe?'] == 'Yes' and row['Priority'] == 'High':
                        colors[analysis_df.columns.get_loc('Safe?')] = 'color: orange'
                    elif row['Safe?'] == 'Yes':
                        colors[analysis_df.columns.get_loc('Safe?')] = 'color: green'
                    return colors

                st.dataframe(
                    analysis_df.style.format({
                        'Latitude': '{:.7f}',
                        'Longitude': '{:.7f}',
                        'Depth (in)': '{:.3f}',
                        'Length (in)': '{:.3f}',
                        'Burst Pressure (psi)': '{:.0f}',
                        'MAOP (psi)': '{:.0f}',
                        'Likelihood': '{:.0f}',
                        'Consequence': '{:.0f}',
                        'Risk Score': '{:.0f}'
                    }).apply(apply_safe_color, axis=1),
                    height=300,
                    width=1100
                )
                st.markdown("""
                    *Notes:*
                    - Red: Unsafe (Burst < MAOP) or Immediate Priority (RPR < 0.2 or failure within 1 year)
                    - Orange: High Priority
                    - Green: Safe/Medium-Low Priority
                    - **Assumption:** Growth rate is assumed (0.01 in/year)
                """)
            else:
                st.info("No metal loss features found in the dataset.")
        
        with col2:
            st.subheader("Inspection Data")
            st.dataframe(
                filtered_df[['Feature Type', 'Feature ID', 'Latitude', 'Longitude', 
                           'RPR Eff. Area 1.25 MAOP']].style.format({
                    'Latitude': '{:.7f}',
                    'Longitude': '{:.7f}',
                    'RPR Eff. Area 1.25 MAOP': '{:.2f}'
                }),
                height=400
            )
            
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_inspection_data.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.error("Error: '20-inch EL100 Pipetally.xlsx' not found in the project folder.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

if __name__ == "__main__":
    main()