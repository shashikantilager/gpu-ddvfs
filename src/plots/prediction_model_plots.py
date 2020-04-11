import matplotlib.pyplot as plt
import numpy as np

SPINE_COLOR = 'gray'
# R2Score, MSE, RMSE, Train_time, Prediction_time
# ---power
catboost = (0.990357645845274, 0.18873697481226268, 0.3841915519706774, 55.978170116742454, 0.015737831592559814)
svr = (-0.10642888454707587, 23.142646392297518, 4.8068960152170686, 0.36597925424575806, 0.12596682707468668)
xgb = (0.981894001175057, 0.3650908092043968, 0.4952977054353822, 0.47278640667597455, 0.004655579725901286)
lr = (-78.40643415209784, 1509.7947859337426, 20.60390399988764, 0.05399475495020548, 0.002534151077270508)
rf = (0.9891727845899833, 0.21247804074038998, 0.3843669420400578, 29.744979560375214, 0.14338640371958414)
lasso = (0.7732857903024044, 4.3421224726440135, 1.9586004758692555, 0.31380218267440796, 0.006018837292989095)
br = (-4.4758661142761715, 104.06590339403375, 7.510301692748501, 0.16240964333216348, 0.003947099049886067)


# -----time`
#
# catboost = (0.9976472428958961,0.003069010742855078,0.04929789961749339, 45.21681731939316,0.023586491743723553)
# svr = (-0.18396903259158018,1.36639865352795,1.1554069173589048, 0.3226270278294881,0.1067923108736674)
# xgb = (0.9971099943173369,0.0036529989681145496,0.05150837357069399, 0.48572323719660443,0.004527648289998372)
# lr = (-3.8609650817578927,4.7820346405405,1.1458855064254732, 0.02688807249069214,0.0018955270449320476)
# rf=  (0.9962601967528193, 0.004872728256138578, 0.05758572563838407, 43.94656995932261, 0.17044216394424438)
# lasso = (0.9756690527494869,0.027043755143609877,0.1626998516719099, 0.23315908511479697,0.004542907079060872)
# br = (-1147.1728829777737,929.0126031947593,15.564238730951528, 0.11601710319519043,0.0033087333043416343)


# https://python-graph-gallery.com/10-barplot-with-number-of-observation/
##USED THIS ONE
# https://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
def rmse_plot():
    N = 1

    ind = np.arange(N)  # the x locations for the groups
    width = 0.4  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 4))

    ##RMSE Old version color
    # rects1 = ax.bar(1, regression[1], width, color='darkblue', edgecolor='black', hatch='xx')
    # rects2 = ax.bar(2, regression_br[1], width, color='blueviolet', edgecolor='black', hatch='oo')
    # rects3 = ax.bar(3, regression_sgd[1], width, color='purple', edgecolor='black', hatch='++')
    # rects4 = ax.bar(4, regression_ridge[1], width, color='tomato', edgecolor='black', hatch='//')
    # rects4 = ax.bar(5, mlp[1], width, color='white', edgecolor='steelblue', hatch='--')
    # rects4 = ax.bar(6, xgb[1], width, color='white', edgecolor='magenta', hatch='..')

    # RMSE New version color

    color = 'white'

    rects1 = ax.bar(1, lr[2], width, color='white', edgecolor='black', hatch='oo')
    # rects3 = ax.bar(2, br[2], width, color=color,  hatch='//')
    rects2 = ax.bar(2, svr[2], width, color='white', edgecolor='black', hatch='++')
    rects3 = ax.bar(3, lasso[2], width, color='white', edgecolor='black', hatch='/')

    rects4 = ax.bar(4, xgb[2], width, color='white', edgecolor='black', hatch='--')
    rects5 = ax.bar(5, catboost[2], width, color='white', edgecolor='black', hatch='xx')

    # # add some text for labels, title and axes ticks
    # ax.set_ylabel("Average Job Duration (seconds)", fontsize=12)
    # # ax.set_title('Scores by group and gender')
    # ax.set_xticks(ind + width * 1.5)
    # ax.set_xticklabels(('WordCount', 'Sort', 'PageRank'), fontsize=10)

    # add some text for labels, title and axes ticks
    ax.set_ylabel("RMSE Values (log scale)", fontsize=12)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_yscale('log')

    ax.set_xticks((1, 2, 3, 4, 5))
    ax.set_xticklabels(('LR', 'SVR', 'Lasso', 'XGBoost', 'CatBoost'), fontsize=10)

    # ax.legend((rects1[0], rects2[0], rects3[0], rects4[0],rects5[0]), ('LR', 'SVR', 'Lasso','XGBoost','CatBoost'), loc='upper left', shadow=True,
    #           fontsize='x-large')

    # create a list to collect the plt.patches data
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
        ax.text(i.get_x() + 0.10, i.get_height(), str(round((i.get_height()), 2)), fontsize=10,
                color='dimgrey')

    # ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('FIFO', 'Morpheus', 'BFD', 'FFD'),loc='upper right', shadow=True,fontsize='large')
    plt.savefig("prediction_models/rmse_power_comparison1.pdf")

    plt.show()


def train_time_plot():
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
    rects2 = ax.bar(2, lr[3], width, color=color, hatch='oo')
    rects3 = ax.bar(3, svr[3], width, color=color, hatch='++')
    rects4 = ax.bar(4, rf[3], width, color=color, hatch='//')
    rects4 = ax.bar(5, xgb[3], width, color=color, hatch='--')
    rects4 = ax.bar(6, catboost[3], width, color=color, hatch='xx')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("RMSE Values", fontsize=14)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks((1, 2, 3, 4, 5))
    ax.set_xticklabels(('LR', 'SVR', 'RF', 'xgb', 'catboost'), fontsize=10)

    # ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('FIFO', 'Morpheus', 'BFD', 'FFD'),loc='upper right', shadow=True,fontsize='large')
    # plt.savefig("prediction_models/training_time.pdf")

    plt.show()


rmse_plot()
