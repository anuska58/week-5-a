import streamlit as st

from apputil import *

# Load Titanic dataset
df=load_data()
st.write(
'''
# Titanic Visualization 1

'''
)
# Generate and display the figure
fig1 = visualize_demographic(df)
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Titanic Visualization 2
'''
)
# Generate and display the figure
fig2 = visualize_families(df)
st.plotly_chart(fig2, use_container_width=True)

st.write(
'''
# Titanic Visualization Bonus
'''
)
# Generate and display the figure
fig3 = visualize_family_size(df)
st.plotly_chart(fig3, use_container_width=True)