
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import numpy as np
import os


def plot_attrition_bar_single(subcategory, col, out_dir):
    """Ordinal - variable"""
    if col not in df.columns:
        print(f"[skip] {col} not in df")
        return

    grp = df.groupby(col)['attrition']
    rate = grp.mean()
    count = grp.count()
    ci = 1.96 * np.sqrt(rate * (1 - rate) / count)

    colors = ['#FF8FA3', '#74C69D', '#4CC9C0', '#A29BFE', '#FDB88A', '#74B9FF']
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(rate)), rate.values,
                  color=colors[:len(rate)], width=0.6, edgecolor='white')

    # error bars
    ax.errorbar(range(len(rate)), rate.values, yerr=ci.values,
                fmt='none', color='black', elinewidth=1.5, capsize=5)
    # mean
    ax.axhline(df['attrition'].mean(), color='black', linestyle='--', linewidth=1.2)

    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{v}\n({count[v]})" for v in rate.index], fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_title(f'Attrition rate by {col}', fontsize=12)
    ax.set_ylim(0, min(1, rate.max() + ci.max() + 0.05))
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f"{subcategory.strip().replace(' ', '_')}_{col}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_attrition_bar_subcat(subcategory, variables_list, out_dir):
    """Ordinal subcategory"""
    valid = [v.replace(' ', '') for v in variables_list if v.replace(' ', '') in df.columns]
    if not valid:
        print(f"[skip] {subcategory}: no valid columns")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Attrition rate - {subcategory}', fontsize=13, fontweight='bold')
    colors = ['#FF8FA3', '#74C69D', '#4CC9C0', '#A29BFE', '#FDB88A', '#74B9FF']

    for ax, col in zip(axes, valid):
        grp = df.groupby(col)['attrition']
        rate = grp.mean()
        count = grp.count()
        ci = 1.96 * np.sqrt(rate * (1 - rate) / count)

        ax.bar(range(len(rate)), rate.values,
               color=colors[:len(rate)], width=0.6, edgecolor='white')
        ax.errorbar(range(len(rate)), rate.values, yerr=ci.values,
                    fmt='none', color='black', elinewidth=1.5, capsize=5)
        ax.axhline(df['attrition'].mean(), color='black', linestyle='--', linewidth=1.2)
        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels([f"{v}\n({count[v]})" for v in rate.index], fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.set_title(col, fontsize=10)
        ax.set_ylim(0, min(1, rate.max() + ci.max() + 0.05))
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f"{subcategory.strip().replace(' ', '_')}_variable.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_attrition_rate_single(subcategory, col, out_dir):
    """variable"""
    if col not in df.columns:
        print(f"[skip] {col} not in df")
        return


    grp = df.groupby(col)['attrition']
    rate = grp.mean()
    count = grp.count()

    # 95% confidence interval
    ci = 1.96 * np.sqrt(rate * (1 - rate) / count)

    # sample size
    labels = [f"{v}\n({count[v]})" for v in rate.index]

    fig, ax = plt.subplots(figsize=(8, max(3, len(rate) * 1.2)))

    colors = ['#4CC9C0', '#FF8FA3']
    for i, (val, r, c) in enumerate(zip(rate.index, rate.values, ci.values)):
        color = colors[i % len(colors)]
        ax.errorbar(r, i, xerr=c, fmt='D', color=color,
                    ecolor=color, elinewidth=2, capsize=6, markersize=8)
        ax.hlines(i, r - c, r + c, color=color, linewidth=2)

    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(df['attrition'].mean(), color='black', linestyle=':', linewidth=1.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.set_title(f'Attrition rate by {col}', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f"{subcategory.strip().replace(' ', '_')}_{col}.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_attrition_rate_subcat(subcategory, variables_list, out_dir):
    """subcategory"""
    valid = [v.replace(' ', '') for v in variables_list if v.replace(' ', '') in df.columns]
    if not valid:
        print(f"[skip] {subcategory}: no valid columns")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Attrition rate - {subcategory}', fontsize=13, fontweight='bold')

    for ax, col in zip(axes, valid):
        grp = df.groupby(col)['attrition']
        rate = grp.mean()
        count = grp.count()
        ci = 1.96 * np.sqrt(rate * (1 - rate) / count)
        labels = [f"{v}\n({count[v]})" for v in rate.index]

        colors = ['#4CC9C0', '#FF8FA3', '#A29BFE', '#FDB88A', '#74B9FF']
        for i, (val, r, c) in enumerate(zip(rate.index, rate.values, ci.values)):
            color = colors[i % len(colors)]
            ax.errorbar(r, i, xerr=c, fmt='D', color=color,
                        ecolor=color, elinewidth=2, capsize=6, markersize=8)
            ax.hlines(i, r - c, r + c, color=color, linewidth=2)

        ax.set_yticks(range(len(rate)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(df['attrition'].mean(), color='black', linestyle=':', linewidth=1.2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax.set_title(col, fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f"{subcategory.strip().replace(' ', '_')}_variable.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_numeric(subcategory, variables_list): 
    valid = [(v, v.replace(' ', '')) for v in variables_list if v.replace(' ', '') in df.columns]
    if not valid:
        print(f"[skip] {subcategory}: no valid columns")
        return

    n = len(valid)
    fig = plt.figure(figsize=(14, 5 * n))
    fig.suptitle(f'Numeric related to {subcategory}', fontsize=14, y=1.01)

    for i, (disp, col) in enumerate(valid):
        gs = gridspec.GridSpec(n, 2, figure=fig)

        att0 = df[df['attrition'] == 0][col].dropna()
        att1 = df[df['attrition'] == 1][col].dropna()
        median_val = df[col].median()

        # left: stacked histogram
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.hist([att0, att1], bins=30, stacked=True,
                 color=['#4CC9C0', '#FF8FA3'], label=['0', '1'],
                 edgecolor='white', linewidth=0.3)
        ax1.axvline(median_val, color='black', linestyle='--', linewidth=1.5)
        ax1.set_title(disp, fontsize=11)
        ax1.set_xlabel(f'Median {disp} is {median_val:.0f}', fontsize=9)
        ax1.legend(title='attrition', fontsize=8)

        # right: density plot
        ax2 = fig.add_subplot(gs[i, 1])
        for grp, color, lc in [(att0, '#4CC9C0', '#00B4AE'), (att1, '#FF8FA3', '#FF4D6D')]:
            if grp.std() == 0:
                print(f"[constant column] {col} - all values are {grp.iloc[0]}")

            if len(grp) > 1 and grp.std() > 0:
                kde = gaussian_kde(grp)
                x = np.linspace(df[col].min(), df[col].max(), 300)
                ax2.fill_between(x, kde(x), alpha=0.5, color=color)
                ax2.plot(x, kde(x), color='black', linewidth=0.8)
                ax2.axvline(grp.mean(), color=lc, linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Lines represent average by group', fontsize=9)
        ax2.legend(['0', '1'], title='attrition', fontsize=8)

    plt.tight_layout()
    fname = f"{subcategory.strip().replace(' ', '_')}_variable.png"
    fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_numeric_seperately(subcategory, variables_list):
    valid = [(v, v.replace(' ', '')) for v in variables_list if v.replace(' ', '') in df.columns]
    if not valid:
        print(f"[skip] {subcategory}: no valid columns")
        return

    for disp, col in valid:
        att0 = df[df['attrition'] == 0][col].dropna()
        att1 = df[df['attrition'] == 1][col].dropna()
        median_val = df[col].median()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Numeric related to {subcategory}', fontsize=14)

        # left: stacked histogram
        ax1.hist([att0, att1], bins=30, stacked=True,
                 color=['#4CC9C0', '#FF8FA3'], label=['0', '1'],
                 edgecolor='white', linewidth=0.3)
        ax1.axvline(median_val, color='black', linestyle='--', linewidth=1.5)
        ax1.set_title(disp, fontsize=11)
        ax1.set_xlabel(f'Median {disp} is {median_val:.0f}', fontsize=9)
        ax1.legend(title='attrition', fontsize=8)

        # right: density plot
        ax2.set_xlabel('Lines represent average by group', fontsize=9)
        for grp, color, lc in [(att0, '#4CC9C0', '#00B4AE'), (att1, '#FF8FA3', '#FF4D6D')]:
            if grp.std() == 0:
                print(f"[constant column] {col} - all values are {grp.iloc[0]}")
                continue
            if len(grp) > 1:
                kde = gaussian_kde(grp)
                x = np.linspace(df[col].min(), df[col].max(), 300)
                ax2.fill_between(x, kde(x), alpha=0.5, color=color)
                ax2.plot(x, kde(x), color='black', linewidth=0.8)
                ax2.axvline(grp.mean(), color=lc, linestyle='--', linewidth=1.5)
        ax2.legend(['0', '1'], title='attrition', fontsize=8)

        plt.tight_layout()
        fname = f"{subcategory.strip().replace(' ', '_')}_{col}.png"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {fname}")
        
if __name__ == "__main__":

    # ── 1. Column names format: lowercase + remove spaces. ──────────────────────────
    path = r"/Users/liyuqiao/Desktop/LYQ/US_academic/academic/NEU/2026Spring/ECON5140/midterm/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df['attrition'] = df['attrition'].map({'Yes': 1, 'No': 0})
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    print(numeric_cols)
    # ── 2. Variables Category ───────────────────────────────────────────────────────
    cat_df = pd.read_excel(r"/Users/liyuqiao/Desktop/LYQ/US_academic/academic/NEU/2026Spring/ECON5140/midterm/Data_categories.xlsx", sheet_name= 'input')

    rows = []
    for _, row in cat_df.iterrows():
        for var in row['Variables'].split(','):
            clean = var.strip().lower().replace('_', ' ').split('(')[0].strip()
            rows.append({
                'variable_type': row['Variable Type'].strip(),
                'subcategory':   row['Subcategory'].strip(),  
                'variable':      clean
            })
    var_df = pd.DataFrame(rows)
    # print(var_df)
    # print(var_df.columns.tolist())
    # ── 3. Create var table ─────────────────────────────────────────────────────────────
    var_df['variable_col'] = var_df['variable'].str.replace(' ', '')
    numeric_var_df = var_df[var_df['variable_col'].isin(numeric_cols)]
    var_table = numeric_var_df.groupby('subcategory')['variable'].apply(list).to_dict()
    # print(var_table)
    # ── 3. Output dir ─────────────────────────────────────────────────


    # ── 4. Go through numeric subcategory and Plot func (conbined) ───────────────────────────────────────────
   
    # out_dir = './midterm/numerical_data_visualization'
    # os.makedirs(out_dir, exist_ok=True)
    # for subcat, vars_str in var_table.items():
    #     plot_numeric(subcat, vars_str)

    # out_dir = './midterm/seperate_numerical_data_visualization'
    # os.makedirs(out_dir, exist_ok=True)
    # for subcat, vars_str in var_table.items():
    #     plot_numeric_seperately(subcat, vars_str)

    # ── 5. Binary and nominal categorical  ───────────────────────────────────────────   
    # out_dir = './midterm/seperate_binary_nominal_data_visualization'
    # os.makedirs(out_dir, exist_ok=True)
    # for _, row in var_df[var_df['subcategory'].isin(['Binary variables', 'Nominal variables'])].iterrows():
    #     plot_attrition_rate_single(row['subcategory'], row['variable_col'], out_dir)

    # out_dir = './midterm/binary_nominal_data_visualization'
    # os.makedirs(out_dir, exist_ok=True)
    # cat_var_table = (var_df[var_df['subcategory'].isin(['Binary variables', 'Nominal variables'])]
    #                 .groupby('subcategory')['variable'].apply(list).to_dict())
    # for subcat, vars_list in cat_var_table.items():
    #     plot_attrition_rate_subcat(subcat, vars_list, out_dir)

    # ── 6. Ordinal categorical  ───────────────────────────────────────────   

    out_dir = './midterm/seperate_ordinal_data_visualization'
    os.makedirs(out_dir, exist_ok=True)
    for _, row in var_df[var_df['variable_type'].isin(['Categorical variables'])].iterrows():
        plot_attrition_bar_single(row['subcategory'], row['variable_col'], out_dir)

    out_dir = './midterm/ordinal_data_visualization'
    os.makedirs(out_dir, exist_ok=True)
    ordinal_var_table = (var_df[var_df['variable_type'] == 'Categorical variables']
                        .groupby('subcategory')['variable'].apply(list).to_dict())
    for subcat, vars_list in ordinal_var_table.items():
        plot_attrition_bar_subcat(subcat, vars_list, out_dir)