import matplotlib.pyplot as plt


def simple_pie_plot(label_list, num_list, title_str):
    """
    Generate a pie plot following a specific type of style

    Args:
        label_list (list of str): list containing str for categories
        num_list (list of int): list containing int for count in each
        categories
        title_str (str):

    Returns:
        0 (int): to indicate successfully run
    """
    tag_list = []  # Create tag_list that will contain info for the categories
    # and the chemicals in each category
    for label, real_num in zip(label_list, num_list):
        tag = label + '\n(' + str(real_num) + ')'
        tag_list.append(tag)

    fig, ax = plt.subplots(figsize=(6, 6))  # Set the figure size
    ax.pie(
        num_list,
        labels=tag_list,
        autopct='%1.0f%%',  # Label each piece of the pie with percentage
        startangle=140,
        textprops={
            'fontsize': 14,
            'fontweight': 'bold',
            'ha': 'center',
            # 'va': 'center'
        }
    )
    plt.axis('equal')  # Ensures that pie is drawn as a circle. Otherwise it
    # can be an oval
    plt.title(title_str)

    return 0
