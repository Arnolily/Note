import matplotlib.pyplot as plt

# Data
categories = ['BP', 'OWM', 'EWC', 'HAT', 'GPM', 'BP*', 'BP**']
values = [63.23, 65.1, 68.34, 74.6, 77.4, 68.1, 76.32]
lower_bounds = [61.96, 63.51, 67.3, 73.6, 77.3, 68.1, 75.82]
upper_bounds = [64.45, 66.79, 69.6, 75.1, 77.6, 68.1, 77.7]
lower_bounds = [values[i] - lower_bounds[i] for i in range(len(values))]
upper_bounds = [upper_bounds[i] - values[i] for i in range(len(values))]

colors = ['#FF5733', '#33C1FF', '#33FF57', '#C70039', '#FFC300', '#8E44AD', '#2ECC71']
# Create figure and axis
fig, ax = plt.subplots()

# Plot bar chart
bars = ax.bar(categories, values, yerr=[lower_bounds, upper_bounds], capsize=5, color=colors)

# Customize the plot
print(plt.style.available)  # List available styles
plt.style.use('ggplot')  # Use a valid style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=10, color='black')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('accuracy')
ax.set_title('Task Increment on Split CIFAR-100')

# Bold specific x-tick labels
xtick_labels = ax.get_xticklabels()
for label in xtick_labels:
    if label.get_text() in ['BP', 'BP*', 'BP**']:
        label.set_fontweight('bold')

# Show plot
plt.show()