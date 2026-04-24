import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from paths import Paths

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    signal = (signal - smin)/(smax - smin)
    return signal

def plot_bold(bold):
    bold = np.array(bold)
    bold = minmaxscale(bold)
    plt.figure(figsize=(6,12))
    plt.plot(range(bold.shape[1])+3*bold, linewidth=0.5)
    plt.show()

def basic_3d_plot(sweep_df, p1_name, 
                        p2_name, p3_name, var_to_plot,
                        ax=None):

    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    x = sweep_df[p1_name].astype(float)
    y = np.log(sweep_df[p2_name].astype(float))
    z = np.log(sweep_df[p3_name].astype(float))
    c = sweep_df[var_to_plot].astype(float)

    if c.max() == c.min():
        sizes = np.ones_like(c) * 10
    else:
        sizes = 1 + (5*(c - c.min()) / (c.max() - c.min()))**4

    # If no axis is provided, create standalone figure
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=sizes, alpha=0.4)

    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_zlabel(p3_name)
    ax.set_title(f'3D Scatter of {var_to_plot}')

    return sc  # return scatter so colorbar can be attached outside

def plot_hist_and_3d(sweep_df, p1_name, p2_name, p3_name, var_to_plot, outpath):

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6))

    # Left: Histogram
    ax1 = fig.add_subplot(1, 2, 1)
    c = sweep_df[var_to_plot].astype(float)
    ax1.hist(c, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_title(f'Distribution of {var_to_plot}')
    ax1.set_xlabel(var_to_plot)
    ax1.set_ylabel("Count")

    # Right: 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc = basic_3d_plot(
        sweep_df, p1_name, p2_name, p3_name, var_to_plot, ax=ax2
    )

    fig.colorbar(sc, ax=ax2, shrink=0.6, label=var_to_plot)

    plt.tight_layout()
    outpath = f'{outpath}/{var_to_plot}_distr.png'
    plt.savefig(outpath)
    plt.close()


def make_tick_formatter(dt=0.5):
    """
    Function for formatting the x-ticks of time expressing them in seconds
    Inputs: the dt of the integration fo the model
    """
    def format_ticks(x, pos):
        return '{:.1f}'.format(x * dt * 1e-3)
    return format_ticks

def plot_eeg(data, channels, dt=0.5):
    """
    Plotting the EEG signals from selected channels
    """
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, len(channels)), sharex=True)

    # Plot each channel's time series in a separate subplot
    for ax_i, ch_i in enumerate(channels):
        axes[ax_i].plot(data[ch_i, :], color='blue', linewidth=0.8)
        axes[ax_i].set_ylabel(f'Ch {ch_i+1}', rotation=0, labelpad=15, fontsize=8)
        axes[ax_i].set_yticks([])
        axes[ax_i].spines['top'].set_visible(False)
        axes[ax_i].spines['right'].set_visible(False)
        axes[ax_i].spines['left'].set_visible(False)
        axes[ax_i].spines['bottom'].set_visible(False)
        axes[ax_i].tick_params(left=False, bottom=False)  # Hide ticks

    # Set common x-label
    axes[-1].xaxis.set_major_formatter(make_tick_formatter(dt))
    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    #plt.savefig(results_path+'first_eeg.png')
    plt.show()

    
def basic_3d_sweep_plot(sweep_df, p1_name, 
    p2_name, p3_name, var_to_plot):
    # Use interactive notebook backend
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    #%matplotlib widget

    # Extract coordinates and variable to plot
    x = sweep_df[p1_name].astype(float)
    y = sweep_df[p2_name].astype(float)
    z = sweep_df[p3_name].astype(float)
    c = sweep_df[var_to_plot].astype(float)

    sizes = 1 + (5*(c - c.min()) / (c.max() - c.min()))**4

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=sizes, alpha=0.4)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(var_to_plot)

    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_zlabel(p3_name)

    plt.title(f'3D Scatter of {var_to_plot}')
    plt.tight_layout()
    plt.show()

def plot_2d_heatmaps(feat_df, title, metrics, columns, index, outpath):
    """
    2d heatmaps created by fixing one value (the pivot) 
    
    :param feat_df: df with data features
    :param metrics: features to be plotted
    :param pivot: the parameter to be fixed in each plot
    :param columns: the param to be used as column of the heatmaps
    :param index: the param to be used as index of the heatmaps
    """

    fig, axes = plt.subplots(
        int(np.sqrt(len(metrics))), int(np.sqrt(len(metrics))),
        figsize=(4 * len(metrics), 4 * len(metrics)),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        dfplot = feat_df.pivot(
            columns=columns,
            index=index,
            values=metric
        )

        im = ax.imshow(
            dfplot.values,
            aspect="auto",
            origin="lower"
        )

        ax.set_title(metric)
        fig.colorbar(im, ax=ax, fraction=0.046)

        # ticks
        ax.set_xticks(np.arange(len(dfplot.columns)))
        ax.set_xticklabels(dfplot.columns, rotation=90)

        ax.set_yticks(np.arange(len(dfplot.index)))
        ax.set_yticklabels(dfplot.index)

        ax.set_xlabel(columns)

    axes[0].set_ylabel(index)
    fig.suptitle(f"{title}", fontsize=14)

    plt.tight_layout()
    savepath = f'{outpath}/{title}_2d_heatmap.png'
    plt.savefig(savepath)
    plt.close()

def save_feat_and_color_by_param(params, scatter0, scatter1, feat_df, outpath):
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, params):

        if param in ['wd', 'ws', 'njdopa_ctx', 'njdopa_str', 'sigma']:
            c = np.log(feat_df[param].astype(float) )  # inverse log10
        else:
            c = feat_df[param].astype(float)

        sc = ax.scatter(
            scatter0,
            scatter1,
            c=c,
            cmap='viridis',
            alpha=0.7
        )

        ax.set_title(f'Colored by {param}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    savepath = f'{outpath}/pca_scatter.png'
    plt.savefig(savepath)
    plt.close()
    #plt.show()

def plot_feat_and_color_by_param(params, scatter0, scatter1, feat_df):
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, param in zip(axes, params):

        if param in ['wd', 'ws', 'sigma']:
            c = np.log(feat_df[param].astype(float) )  # inverse log10
        else:
            c = feat_df[param].astype(float)

        sc = ax.scatter(
            scatter0,
            scatter1,
            c=c,
            cmap='viridis',
            alpha=0.7
        )

        ax.set_title(f'Colored by {param}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_pca_feat_importance(params, X_r, feat_df, importance):

    pc1_corr = [
        np.corrcoef(X_r[:, 0], feat_df[p])[0, 1]
        for p in params]

    pc2_corr = [
        np.corrcoef(X_r[:, 1], feat_df[p])[0, 1]
        for p in params]

    importance_params = importance.loc[
        importance.index.intersection(params)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    # LEFT: Importance
    axes[0].bar(params, importance_params.values)
    axes[0].set_title("Feature Importance")
    axes[0].set_ylabel("Absolute Loading")

    # RIGHT: Correlation with PC1
    axes[1].bar(params, pc1_corr)
    axes[1].set_title("Correlation with PC1")
    axes[1].set_ylabel("Pearson r")
    # RIGHT: Correlation with PC1
    axes[2].bar(params, pc2_corr)
    axes[2].set_title("Correlation with PC1")
    axes[2].set_ylabel("Pearson r")

    plt.tight_layout()
    plt.show()


def plot_sbi_violin_estimated_params(params_label, est_params, path_to_save, ses, type_of_sweep, type_of_extraction):
    """
    Plot the violin plots of the estimated parameters distribution (posterior distribution from sbi)
    """
    p1, p2, p3 = est_params
    plt.figure(figsize=(10, 3))
    for i, variables in enumerate([p1, p2, p3]):
        plt.subplot(1,params_label.shape[0],i+1)
        plt.violinplot(variables, widths=0.7, showmeans=True, showextrema=True);
        #plt.axhline(y=true_params[i], linewidth=2, color='r')
        plt.ylabel(str(params_label[i]), fontsize=24)   
        plt.xticks([])
        plt.yticks(fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(f"{path_to_save}/{ses}_{type_of_sweep}_{type_of_extraction}_PosteriorViloin_EstimatedParams_obs.png", dpi=300)
    plt.close()

def plot_sbi_kde_distr(params_label, prior, posterior_samples, path_to_save, ses, type_of_sweep, type_of_extraction):
    plt.figure(figsize=(16, 4))

    for i in range (len(params_label)): 
        prior_samples=np.stack([prior.sample().tolist() for _ in range(1000)], axis=0)
        ax = plt.subplot(1, 3, i + 1)
        ax=sns.distplot(posterior_samples[:,i], bins=100, hist=False, kde=True, rug=False, rug_kws={"alpha": 0.1, "color": "g"},
                    kde_kws={"color": "b", "alpha": 0.9, "lw": 3, "label": "Posterior"})
        
        ax_=sns.distplot(prior_samples[:,i], bins=100, hist=False, kde=True, rug=False, rug_kws={"alpha": 0.1, "color": "g"},
                    kde_kws={"color": "g", "alpha": 0.9, "lw": 3, "label": "Prior"})

        x_ = ax.lines[0].get_xdata()
        y_ = ax.lines[0].get_ydata()
        ymax = max(y_)
        xpos = np.argmax(y_, axis=0)
        xmax = x_[xpos]
        plt.vlines(x=xmax, ymin=0., ymax=y_.max(), colors='cyan', label='MAP')
        #plt.vlines(x=true_params[i], ymin=0., ymax=y_.max(), colors='r', label='Truth')

        plt.xlabel(str(params_label[i]), fontsize=20)   
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if i==0:
            plt.ylabel(' Posterior ', fontsize=18);  
        else:
            plt.ylabel('')
            
        if i==2:
                plt.legend(fontsize=16, frameon=False)
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f"{path_to_save}/{ses}_{type_of_sweep}_{type_of_extraction}_Posterior_EstimatedParams_kde.png", dpi=300)
    plt.close()

def plot_signal_and_matrices(pid, ses, combination, filtered_bold, fcd, var_fcd, fc, mean_fc, path):

    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    bold_max = filtered_bold.max(axis=0)
    bold_min = filtered_bold.min(axis=0)
    plt.plot(2*((filtered_bold-bold_min)/(bold_max-bold_min)) + np.arange(filtered_bold.shape[1]), linewidth=0.5)
    plt.title("Filtered BOLD signals")
    plt.subplot(132)
    plt.imshow(fcd, cmap='viridis')
    plt.title(f"FCD, VAR={var_fcd:.5f}")
    plt.subplot(133)
    plt.imshow(fc, cmap='viridis')
    plt.title(f"FC, GBC={mean_fc:.5f}")
    plt.suptitle(f"Subject {pid}, session {ses}, strategy: {combination}")
    plt.savefig(f"{path}/{pid}_{ses}_{combination}_signal_matrices.png")
    plt.close()

def basic_3d_sweep_plot_with_planes_for_ppc(sweep_df, ppc_df, p1_name, 
                                    p2_name, p3_name, var_to_plot, vars_x_dic, cmap, save_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    # ---------------------------------------
    # Extract data
    # ---------------------------------------
    x = sweep_df[p1_name].astype(float).values
    y = sweep_df[p2_name].astype(float).values
    z = sweep_df[p3_name].astype(float).values
    c = sweep_df[var_to_plot].astype(float).values

    l = ppc_df[p1_name].astype(float).values
    m = ppc_df[p2_name].astype(float).values
    n = ppc_df[p3_name].astype(float).values

    # ---------------------------------------
    # Point sizes scaled by variable
    # ---------------------------------------
    if c.max() == c.min():
        sizes = np.ones_like(c) * 10
    else:
        sizes = 1 + (5*(c - c.min()) / (c.max() - c.min()))**4

    # ---------------------------------------
    # Create figure
    # ---------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=sizes, alpha=0.5, edgecolors='none')
    ax.scatter(l, m, n, c='r', alpha=1, s=20)

    # ---------------------------------------
    # Interpolation grids for planes
    # ---------------------------------------
    xi = np.linspace(x.min(), x.max(), 40)
    yi = np.linspace(y.min(), y.max(), 40)
    zi = np.linspace(z.min(), z.max(), 40)

    XI, YI = np.meshgrid(xi, yi)
    XI2, ZI = np.meshgrid(xi, zi)
    YI2, ZI2 = np.meshgrid(yi, zi)

    # Interpolate values for contour planes
    C_xy = griddata((x, y), c, (XI, YI), method='linear')
    C_xz = griddata((x, z), c, (XI2, ZI), method='linear')
    C_yz = griddata((y, z), c, (YI2, ZI2), method='linear')

    # ---------------------------------------
    # Add padding
    # ---------------------------------------
    pad = 0.15
    xr, yr, zr = x.max() - x.min(), y.max() - y.min(), z.max() - z.min()
    xmin, xmax = x.min() - pad*xr, x.max() + pad*xr
    ymin, ymax = y.min() - pad*yr, y.max() + pad*yr
    zmin, zmax = z.min() - pad*zr, z.max() + pad*zr
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # ---------------------------------------
    # Contour planes on cube walls
    # ---------------------------------------
    ax.contourf(XI, YI, C_xy, zdir='z', offset=zmin, cmap=cmap, levels=15, alpha=0.3)
    ax.contourf(XI2, C_xz, ZI, zdir='y', offset=ymax, cmap=cmap, levels=15, alpha=0.3)
    ax.contourf(C_yz, YI2, ZI2, zdir='x', offset=xmin, cmap=cmap, levels=15, alpha=0.3)

    # ---------------------------------------
    # Labels and colorbar
    # ---------------------------------------
    ax.set_xlabel(vars_x_dic[p1_name])
    ax.set_ylabel(vars_x_dic[p2_name])
    ax.set_zlabel(vars_x_dic[p3_name])
    ax.set_title(f'3D Scatter of {var_to_plot}')

    cbar = fig.colorbar(sc, shrink=0.6, pad=0.1, label=var_to_plot)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.5, labelsize=10)

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}.png')
    plt.close()

def save_feat_and_color_by_param_for_ppc(params, sweep_r, ppc_r, emp_r, feat_df, outpath):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5))

    # If only one param, axes is not iterable
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):

        # --- Color for sweep points ---

        c = feat_df[param].astype(float)

        # --- Sweep (background cloud) ---
        sc = ax.scatter(
            sweep_r[:, 0],
            sweep_r[:, 1],
            c=c,
            cmap='viridis',
            alpha=0.5,
            s=20,
            label='Sweep'
        )

        # --- PPC (red points) ---
        ax.scatter(
            ppc_r[:, 0],
            ppc_r[:, 1],
            color='red',
            alpha=0.8,
            s=40,
            label='PPC'
        )

        # --- Empirical (big black dot) ---
        ax.scatter(
            emp_r[:, 0],
            emp_r[:, 1],
            color='black',
            s=120,
            edgecolor='white',
            linewidth=1.5,
            label='Empirical',
            zorder=5
        )

        ax.set_title(f'Colored by {param}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.colorbar(sc, ax=ax)
        ax.legend()

    plt.tight_layout()
    savepath = f'{outpath}.png'
    plt.savefig(savepath, dpi=300)
    plt.close()

def plot_med_results(y, result_df, pid, remission, save_path):
    result_df['med_zi_norm'] = result_df.groupby('medication')['med_zi'].transform(lambda x: x - x.min() + 1e-3)

    import seaborn as sns
    import matplotlib.pyplot as plt

    cmap = sns.color_palette("viridis", as_cmap=True)

    point_color = cmap(0.6)  
    line_color = cmap(0.7)

    g = sns.FacetGrid(
        result_df,
        col="medication",
        col_wrap=2,
        sharey=True,
        height=4
    )

    g.map_dataframe(
        sns.scatterplot,
        x="med_zi_norm",
        y=y,
        alpha=0.007,
        color=point_color
    )

    g.map_dataframe(
        sns.lineplot,
        x="med_zi_norm",
        y=y,
        estimator="mean",
        ci=None,
        color=line_color
    )

    for ax in g.axes.flatten():
        #ax.set_xlim(0, 19)
        #ax.set_xticks(range(20))
        ax.set_xlim(left=0.5*1e0)
        ax.set_xscale('log')

    g.set_axis_labels("med_zi (log scale)", "Improvement score")

    plt.suptitle(f'{pid}, remission: {remission}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_feat_and_color_by_param_for_med(params, medication, sweep_r, med_r, base_emp_r, fup_emp_r, sweep_df, remission, outpath):
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis
    purple = cmap(0.1)   # dark purple
    green = cmap(0.6)    # green
    yellow = cmap(0.95)  # yellow

    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5))

    # If only one param, axes is not iterable
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):

        # --- Color for sweep points ---

        c = sweep_df[param].astype(float)

        # --- Sweep (background cloud) ---
        sc = ax.scatter(
            sweep_r[:, 0],
            sweep_r[:, 1],
            c=c,
            cmap='viridis',
            alpha=0.5,
            s=20,
            label='Sweep'
        )

        # --- Medication (yellow points) ---
        ax.scatter(
            med_r[:, 0],
            med_r[:, 1],
            color='orange',
            alpha=0.8,
            s=40,
            label=medication
        )

        # --- Empirical (big purple dot) ---
        ax.scatter(
            base_emp_r[:, 0],
            base_emp_r[:, 1],
            color=purple,
            s=120,
            edgecolor='white',
            linewidth=1.5,
            label='Baseline',
            zorder=5
        )

        # --- Medication empirical (big green dot) ---
        ax.scatter(
            fup_emp_r[:, 0],
            fup_emp_r[:, 1],
            color=green,
            s=120,
            edgecolor='white',
            linewidth=1.5,
            label='Follow-up',
            zorder=5
        )

        ax.set_title(f'Colored by {param} for {medication}, remission: {remission}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.colorbar(sc, ax=ax)
        ax.legend()

    plt.tight_layout()
    savepath = f'{outpath}'
    plt.savefig(savepath, dpi=300)
    plt.close()

def plot_pca_with_base_med_and_emp(medication, sweep_r, med_r, ppc_r, base_emp_r, fup_emp_r, med_df, remission, outpath):
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis
    purple = cmap(0.1)   # dark purple
    green = cmap(0.6)    # green
    yellow = cmap(0.95)  # yellow

    fig = plt.figure(figsize=(6, 5))

    # --- Color for sweep points ---

    c = med_df['med_zi'].astype(float)

    # --- Sweep (background cloud) ---
    sc = ax.scatter(
        sweep_r[:, 0],
        sweep_r[:, 1],
        c=c,
        cmap='viridis',
        alpha=0.5,
        s=20,
        label='Sweep'
    )

    # --- Medication (yellow points) ---
    ax.scatter(
        med_r[:, 0],
        med_r[:, 1],
        color='orange',
        alpha=0.8,
        s=40,
        label=medication
    )

    # --- Empirical (big purple dot) ---
    ax.scatter(
        base_emp_r[:, 0],
        base_emp_r[:, 1],
        color=purple,
        s=120,
        edgecolor='white',
        linewidth=1.5,
        label='Baseline',
        zorder=5
    )

    # --- Medication empirical (big green dot) ---
    ax.scatter(
        fup_emp_r[:, 0],
        fup_emp_r[:, 1],
        color=green,
        s=120,
        edgecolor='white',
        linewidth=1.5,
        label='Follow-up',
        zorder=5
    )

    ax.set_title(f'Colored by {param} for {medication}, remission: {remission}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    fig.colorbar(sc, ax=ax)
    ax.legend()

    plt.tight_layout()
    savepath = f'{outpath}'
    plt.savefig(savepath, dpi=300)
    plt.close()