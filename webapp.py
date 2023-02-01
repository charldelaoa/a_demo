import streamlit as st
from bokeh.plotting import figure, show
import numpy as np
import pandas as pd

st.set_page_config(page_title="Super-Gaussian Equation Plotter", layout="wide")
# Define the default values for the variables
mu = st.sidebar.slider("Mean", -5.0, 5.0, 0.0, 0.1)
sigma = st.sidebar.slider("Standard Deviation", 0.1, 5.0, 1.0, 0.1)
n = st.sidebar.slider("Order", 2, 10, 2, 1)

st.sidebar.markdown("This is a checkbox")
agree = st.sidebar.checkbox('I agree')

st.sidebar.markdown("Multiselect")
options = st.sidebar.multiselect(
    'What are your favorite colors',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Red'])

st.sidebar.write('You selected:', options)


numeric = st.sidebar.number_input("numerical input", 1, 4, 1, 1)
st.sidebar.write(f"Numeric input modified {numeric}")
# Create a function to plot the equation
def plot_equation(mu, sigma, n):
    x = np.linspace(-5, 5, 1000)
    y = np.exp(-((x-mu)/sigma)**n)
    p = figure(title="Super-Gaussian", x_axis_label='x', y_axis_label='y', width = 300, height= 300)
    p.line(x, y, line_width=2)
    st.bokeh_chart(p)

# Create the Streamlit app

st.title("Super-Gaussian Equation Plotter")

plot_equation(mu, sigma, n)

#####
st.title("This is a dataframe")
df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))
st.dataframe(df)  # Same as st.write(df)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)


