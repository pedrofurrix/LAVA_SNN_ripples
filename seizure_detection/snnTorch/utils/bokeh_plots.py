# Interactive Plot for the spiking events
# bokeh docs: https://docs.bokeh.org/en/2.4.1/docs/first_steps/first_steps_1.html
import bokeh.plotting as bplt
from bokeh.io import curdoc
from bokeh.models import BoxAnnotation, Whisker, ColumnDataSource, Range1d, BasicTickFormatter
import numpy as np
from math import ceil

# Apply the theme to the plot
curdoc().theme = "caliber"  # Can be one of "caliber", "dark_minimal", "light_minimal", "night_sky", "contrast"

# colors list
line_colors = ["green", "blue", "red", "orange", "purple", "cyan", "pink", "brown", "mediumaquamarine", "teal", "olive", "darkgreen", "black", 
               "mediumslateblue", "lightsalmon", "gold", "indigo", "aqua", "rosybrown", "chocolate"]  # TODO: Add more colors if needed

'''
Color Map for different BoxAnnotations
'''
box_color_map = {                  
    'Spike': "#fa0000",                     # 'red',
    'Fast-Ripple': "#0000ff",               # 'blue',
    'Ripple': "#0000ff",                    # 'green',  
    'Spike+Ripple': "#ffe600",              # 'yellow',
    'Spike+Fast-Ripple': "#ff1ffb",         # 'pink',
    'Ripple+Fast-Ripple': "#00f7ff",        # 'cyan',
    'Spike+Ripple+Fast-Ripple': "#000000",  # 'black'
}


""" Only possible in python>= 3.12 
type BoxAnnotationParams = {
    "bottom": float,
    "top": float,
    "left": float,
    "right": float,
    "fill_alpha": float,
    "fill_color": str
} """

"""
create_line_plot: Create a figure with the given parameters
Args:
    title (str): Title of the plot
    x_axis_label (str): Label of the x-axis
    y_axis_label (str): Label of the y-axis
    x (np.ndarray): Array of x values
    y_arrays (list): List of tuples containing the y values and the legend label
        - Shape: [(y_values, legend_label), ...]. Example: [(y_array_1, "Legend 1"), (y_array_2, "Legend 2")]
    x_range (tuple): Range of the x-axis    (Can make linked ranges with other plots)
    y_range (tuple): Range of the y-axis
    sizing_mode (str): Sizing mode of the plot
    tools (str): Tools to be added to the plot
    tooltips (str): Tooltips to be added to the plot
    legend_location (str): Location of the legend
    legend_bg_fill_color (str): Background fill color of the legend
    legend_bg_fill_alpha (float): Background fill alpha of the legend
    box_annotation_params (dict): Parameters to create a box annotation
Returns:
    bplt.Figure: The plot
"""
def create_line_plot(title: str, x_axis_label: str, y_axis_label: str, 
               x: np.ndarray, y_arrays: np.ndarray[tuple[np.ndarray, np.ndarray]],
               x_range=None, y_range=None,
               sizing_mode=None, tools=None, tooltips=None, 
               legend_location=None, legend_bg_fill_color=None, legend_bg_fill_alpha=None, 
               box_annotation_params=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools=tools or "pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        tooltips=tooltips or "Data point @x: @y",
    )

    # Set the range of the x and y-axis
    if x_range is not None:
        p.x_range = x_range
    if y_range is not None:
        p.y_range = y_range

    # Add a line graph to the plot for each y_array
    for (arr_idx, y_array) in enumerate(y_arrays):
        p.line(x, y_array[0], legend_label=y_array[1], line_width=2, line_color=line_colors[arr_idx % len(line_colors)])

    # Legend settings
    p.legend.location = legend_location or "top_right"
    p.legend.background_fill_color = legend_bg_fill_color or "navy"
    p.legend.background_fill_alpha = legend_bg_fill_alpha or 0.1
    p.legend.click_policy = "hide"  # Clicking on a legend item will hide the corresponding line
    # Format legend to 2 columns
    p.legend.ncols = ceil(len(y_arrays) / 7)    # Make the number of rows no more than 7

    # Grid settings
    # p.ygrid.grid_line_color = "red"

    # Add a box annotation
    if box_annotation_params is not None:
        inner_box = BoxAnnotation(
            bottom=box_annotation_params["bottom"], 
            top=box_annotation_params["top"], 
            left=box_annotation_params["left"], 
            right= box_annotation_params["right"], 
            fill_alpha=box_annotation_params["fill_alpha"], 
            fill_color=box_annotation_params["fill_color"],
        )

        p.add_layout(inner_box)

    # Change the number of decimal places on hover
    p.hover.formatters = {'@x': 'numeral', '@y': 'numeral'}
    p.hover.tooltips = [("x", "@x{0.0}"), ("y", "@y{0.0000}")]

    # Return the plot
    return p

"""
create_raster_plot: Create a raster figure with the given parameters

Args:
    title (str): Title of the plot
    x_axis_label (str): Label of the x-axis
    y_axis_label (str): Label of the y-axis
    x (np.ndarray): Array of x values
    y_arrays (list): List of tuples containing the (x, y) pairs for each dot to be plotted
    sizing_mode (str): Sizing mode of the plot
    tools (str): Tools to be added to the plot
    tooltips (str): Tooltips to be added to the plot
    legend_location (str): Location of the legend
    legend_bg_fill_color (str): Background fill color of the legend
    legend_bg_fill_alpha (float): Background fill alpha of the legend
    box_annotation_params (dict): Parameters to create a box annotation
Returns:
    bplt.Figure: The plot
"""
def create_raster_plot(title, x_axis_label, y_axis_label, 
               x, y, dot_size=10, sizing_mode=None, tools=None, 
               tooltips=None, box_annotation_params=None, y_axis_ticker=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools=tools or "pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        tooltips=tooltips or "Data point @x: @y"
        # y_range=(-0.5, 5)  # Set the range of the y-axis,
    )

    # Add the dots to the plot
    p.dot(x, y, size=dot_size, color="blue", alpha=1)

    # Axis settings
    p.xaxis.formatter = BasicTickFormatter(use_scientific=False)

    # Set y-axis to integers
    if y_axis_ticker is not None:
        p.yaxis.ticker = y_axis_ticker


    # Grid settings
    # p.ygrid.grid_line_color = "red"

    # Add a box annotation
    if box_annotation_params is not None:
        inner_box = BoxAnnotation(
            bottom=box_annotation_params["bottom"], 
            top=box_annotation_params["top"], 
            left=box_annotation_params["left"], 
            right= box_annotation_params["right"], 
            fill_alpha=box_annotation_params["fill_alpha"], 
            fill_color=box_annotation_params["fill_color"]
        )
        p.add_layout(inner_box)

    # Change the number of decimal places on hover
    p.hover.formatters = {'@x': 'numeral', '@y': 'numeral'}
    p.hover.tooltips = [("x", "@x{0.0}"), ("y", "@y{0.0000}")]

    # Return the plot
    return p


"""
create_raster_fig: Create a raster figure with the given parameters

Args:
    title (str): Title of the plot
    x_axis_label (str): Label of the x-axis
    y_axis_label (str): Label of the y-axis
    x (np.ndarray): List containing the labels for each Bar
    y (list): List containing the counts of each Bar
    x_range (tuple): Range of the x-axis    (Can be used to sort the bars in a specific order)
    sizing_mode (str): Sizing mode of the plot
    bar_width (float): Width of the bars
    is_vertical (bool): If the bars are vertical or horizontal
    height (int): Height of the plot
Returns:
    bplt.Figure: The plot
"""
def create_bar_fig(title, x_axis_label, y_axis_label, 
               x, y, x_range, sizing_mode=None, tooltips=None, 
               bar_width=0.9, is_vertical=True, height=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_range=x_range,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools="pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        tooltips=tooltips or "Data point @x: @top"
        # y_range=(-0.5, 5)  # Set the range of the y-axis,
    )

    # Add the bars to the plot
    if is_vertical:
        p.vbar(x=x, top=y, width=bar_width, color="blue")
    else:
        p.hbar(y=x, right=y, height=bar_width, color="blue")

    # Set the height of the plot
    if height is not None:
        p.plot_height = height

    # Axis settings
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    # Return the plot
    return p


def create_histogram(title, x, bins, x_range=None, legend_label="Histogram", is_density=False, 
                     sizing_mode=None, x_axis_label = 'x', y_axis_label = "y", height=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools="pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        # y_range=(-0.5, 5)  # Set the range of the y-axis,
    )

    # Histogram
    hist, edges = np.histogram(x, density=is_density, bins=bins)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white",
           fill_color="blue", legend_label=legend_label)

    # Set the height of the plot
    if height is not None:
        p.plot_height = height

    # Axis settings
    p.y_range.start = 0
    if x_range is not None:
        p.x_range = x_range

    # Return the plot
    return p


def create_box_plot(title, box_arrays: np.ndarray, sizing_mode=None, 
                    x_axis_labels = None, y_axis_label = "y", height=None):
    """
    create_box_plot: Create a box plot with the given parameters
    @param title: Title of the plot
    @param box_arrays: List of arrays containing the values for each box
    @param sizing_mode: Sizing mode of the plot
    @param x_axis_label: Label of the x-axis
    @param y_axis_label: Label of the y-axis
    @param height: Height of the plot
    """

    # Create the plot
    p = bplt.figure(title=title, 
            background_fill_color="white", y_axis_label=y_axis_label)

    # Define the number of boxes
    n_boxes = len(box_arrays)

    # Store the max_upper value
    max_upper = 0

    quantiles = [0.25, 0.5, 0.75]
    for (box_idx, box_array) in enumerate(box_arrays):
        # Convert the box array to a numpy array
        box_array = np.array(box_array)

        # Calculate the quantiles of the ripple max amplitudes
        ripple_quantiles = np.quantile(box_array, quantiles)

        print(f"ripple_quantiles:  {ripple_quantiles}")

        # Compute the IQR outlier boundaries
        ripple_iqr = ripple_quantiles[2] - ripple_quantiles[0]
        print("Ripple IQR: ", ripple_iqr)

        # Calculate the upper and lower whisker values
        upper = ripple_quantiles[2] + 1.5 * ripple_iqr
        lower = max(ripple_quantiles[0] - 1.5 * ripple_iqr, 0)
        # Update the max_upper value
        max_upper = max(max_upper, upper)

        # Calculate the x-offset for the box
        x_offset = box_idx * 1.0
        # Calculate the position of the box
        x_left = 0.4 + x_offset
        x_right = 0.6 + x_offset

        # Add a box annotation for the IQR of the current box
        box = BoxAnnotation(top=ripple_quantiles[2], bottom=ripple_quantiles[0], left=x_left, right=x_right, 
                            fill_color="green", fill_alpha=0.4,
                            line_color="black", line_alpha=1.0, line_width=1.5)
        p.add_layout(box)

        # Add a line for the median
        p.line([x_left, x_right], [ripple_quantiles[1], ripple_quantiles[1]], line_color="black", line_width=2)

        # Add whiskers
        source = ColumnDataSource(data=dict(values=box_array))
        # Calculate the x-position of the whisker
        x_whisker = 0.5 + x_offset
        upper_whisker = Whisker(source=source, base=x_whisker, upper=upper, lower=lower)
        p.add_layout(upper_whisker)

        # Add Outliers
        # Find the outliers
        upper_outliers = box_array[box_array > upper]
        lower_outliers = box_array[box_array < lower]
        p.circle([x_whisker] * len(upper_outliers), upper_outliers, size=5, color="red", fill_alpha=0.6)
        p.circle([x_whisker] * len(lower_outliers), lower_outliers, size=5, color="red", fill_alpha=0.6)

    # Change the axis
    p.x_range = Range1d(0, n_boxes)

    # Calculate the y_min value according to the max_upper value
    y_min_range = -max(0.2, max_upper * 0.05)
    p.y_range = Range1d(y_min_range, max_upper * 1.2)

    # Add the x-axis labels
    x_axis_ticks = [0.5 + i for i in range(n_boxes)]
    p.xaxis.ticker = x_axis_ticks
    if x_axis_labels is not None:
        # Override the x-axis labels
        p.xaxis.major_label_overrides = dict(zip(x_axis_ticks, x_axis_labels))

    # Return the plot
    return p
