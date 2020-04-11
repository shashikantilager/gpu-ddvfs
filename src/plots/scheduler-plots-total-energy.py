import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dir =  "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/schedule-data/"
# job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_30/"

dir = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data/"
# file= dir +  "job_level_dmon_avg_50_alpha_2.csv"
# file = dir +"job_level_dmon_avg_50.csv"
file = dir + "job_level_dmon_avg_kmeans_w_50_RD.csv"


def energy_bar_plot():
    df = pd.read_csv(file)
    N = 1

    ind = np.arange(N)  # the x locations for the groups
    width = 0.40  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 4))

    ##RMSE Old version color
    # rects1 = ax.bar(1, regression[1], width, color='darkblue', edgecolor='black', hatch='xx')
    # rects2 = ax.bar(2, regression_br[1], width, color='blueviolet', edgecolor='black', hatch='oo')
    # rects3 = ax.bar(3, regression_sgd[1], width, color='purple', edgecolor='black', hatch='++')
    # rects4 = ax.bar(4, regression_ridge[1], width, color='tomato', edgecolor='black', hatch='//')
    # rects4 = ax.bar(5, mlp[1], width, color='white', edgecolor='steelblue', hatch='--')
    # rects4 = ax.bar(6, xgb[1], width, color='white', edgecolor='magenta', hatch='..')

    # RMSE New version color

    color = 'gray'
    ##power
    apps = df.App
    barWidth = 0.4
    y = [df.Max.sum(), df.Default.sum(), df.QEAware.sum()]
    rects1 = ax.bar(1, y[0], width=barWidth, color='white', edgecolor='black', hatch='++')
    rects2 = ax.bar(2, y[1], width=barWidth, color='white', edgecolor='black', hatch='oo')
    # rects3 = ax.bar(3, rf[2], width, color=color,  hatch='//')
    rects4 = ax.bar(3, y[2], width=barWidth, color='white', edgecolor='black', hatch='xx')

    # plt.bar(r1, bars1, color='lightcoral', width=barWidth, edgecolor='black', label='Max Clock', hatch ="//")
    # plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='black', label='Default Clock',hatch ="--")
    # plt.bar(r3, bars3, color='c', width=barWidth, edgecolor='black', label='D-DVFS', hatch ="++")

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Average total power (Watts)", fontsize=14)
    ax.set_xlabel("Policies", fontsize=14)
    # ax.set_yscale('log')

    ax.set_xticks((1, 2, 3))
    ax.set_xticklabels(('MC', 'DC', 'D-DVFS'), fontsize=12)

    # create a list to collect the plt.patches data ---------------------------------------------
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        # use + or - to adjust the postion as shown below
        # ax.text(i.get_x() + 0.10, i.get_height() + 0.04, str(round((i.get_height()), 2)), fontsize=10,
        ax.text(i.get_x() + 0.10, i.get_height() + 0.09, str(round((i.get_height()), 2)), fontsize=12,
                color='black')

    plt.savefig(
        "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data-schedule-energy/scheduling_power_kmeans_total_w50_RD_bw.pdf",
        bbox_inches='tight')

    plt.show()


energy_bar_plot()
