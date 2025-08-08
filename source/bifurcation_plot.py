import matplotlib.pyplot as plt

CATEGORY_LABELS = {
    0: "Chaos (a)",
    1: "Chaos and Cyclic Points (b)",
    2: "Quasi-Periodic Orbit (c)",
    3: "Quasi-Periodic Orbit and Cyclic Points (d)",
    4: "Multi-Year Cycles (e)",
    5: "One Year Cycle (f)"
}

def plot_bifurcation_diagram(a_s_unique, nu_unique, category_matrix, save_path=None):
    """
    This function uses the results from the classification to plot the 
    bifurcation diagram. 

    Args:
        a_s_unique (array): Unique values of a_s.
        nu_unique (array): Unique values of nu.
        category_matrix (2D array): Matrix of category values.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    # Create a colormap with a fixed number of categories
    cmap_categories = plt.get_cmap('rainbow', len(CATEGORY_LABELS))
    # Set font properties
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    # Create the plot with squares/rectangles filling out the whole space
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    plt.pcolormesh(a_s_unique, nu_unique, category_matrix, cmap=cmap_categories, shading='auto', vmin=0, vmax=len(CATEGORY_LABELS) - 1)
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label("Category Descriptions", fontsize=8, labelpad=5)
    cbar.set_ticks(range(len(CATEGORY_LABELS)))
    cbar.set_ticklabels(CATEGORY_LABELS.values())
    # Add labels and title
    plt.xlabel(r'$a_s$', fontsize=12)
    plt.ylabel(r'$\nu$', fontsize=12)
    plt.title("Numerical Bifurcation Diagram", fontsize=10, pad=5)
    # Adjust layout
    plt.tight_layout()
    # Save the plot if a save path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    # Show the plot
    plt.show()