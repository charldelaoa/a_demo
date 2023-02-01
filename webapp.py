import streamlit as st
from bokeh.plotting import figure, show
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from bokeh.plotting import figure, show

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

st.title("Generate interactive HTML plots")
st.line_chart(chart_data)

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.area_chart(chart_data)

x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)



N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = np.array([ [r, g, 150] for r, g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8")

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"

p = figure(tools=TOOLS)

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)

st.bokeh_chart(p)

