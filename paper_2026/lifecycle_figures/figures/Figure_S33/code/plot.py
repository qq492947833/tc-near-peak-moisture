from display_format import format_number,format_group
MASTER_SHA = '0b4c4308543fb4e7f5f32911d6b654a5fc4e720b8617fc53ce46a59b2c117b37'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
TEMP = BLUE = '#167B9E'
FINAL = ORANGE = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
GREEN = '#418D79'
RINGS = ['100_300km', '300_800km']
RINGLABELS = ['100–300 km', '300–800 km']
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Deep shear', 'Mid-level RH', 'Divergence', 'Ascent', 'Vorticity']
BS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
BC = dict(zip(BS, ['#437F9C', '#BD9145', '#428C79', '#9D82AB', '#AB685E', '#67A4AC']))
BASINS = BS
WIDTH_MM = 180
WIDTH_IN = 180 / 25.4
GRID = '#DDE3E7'
SCOPES = ['main_all', 'exclude_single_6h', 'strict_bridge_tropical', 'without_Freddy2023', 'without_basin_WP', 'without_basin_EP', 'without_basin_SI', 'without_basin_NA', 'without_basin_SP', 'without_basin_NI']

def main():
    annual = source('coverage/annual').set_index('genesis_year')['n_sid']
    monthly = source('coverage/month_by_basin_percent').set_index('basin_genesis')
    fraction = source('coverage/annual_basin_fraction').set_index('genesis_year')
    support = source('coverage/stage_environment_support').set_index('variable')
    for tag, dat in [('a', annual.rename('n_sid').reset_index()), ('b', monthly.reset_index()), ('c', fraction.reset_index()), ('d', support.reset_index())]:
        table('Coverage', tag, dat, 'frozen_coverage_display')
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    fig, ax = plt.subplots(2, 2, figsize=(180 / 25.4, 160 / 25.4))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.91, bottom=0.15, wspace=0.52, hspace=0.68)

    def title(a, letter, t):
        a.set_gid(letter)
        a.set_title(t, loc='left', pad=12)
        a.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-19, 12), textcoords='offset points', weight='bold', fontsize=10, va='bottom')
    a = ax[0, 0]
    a.plot(annual.index, annual.values, color=TEMP, lw=1.4)
    a.fill_between(annual.index, 0, annual.values, color=TEMP, alpha=0.1)
    a.set(xlabel='Genesis year', ylabel='Master cyclones', xlim=(1981.5, 2023.5), ylim=(0, None), xticks=[1982, 1990, 2000, 2010, 2023])
    title(a, 'a', 'Master sample')
    a = ax[0, 1]
    im = a.imshow(monthly.to_numpy(), aspect='auto', cmap='Blues', vmin=0, vmax=40)
    a.set(xticks=np.arange(12), xticklabels=range(1, 13), yticks=np.arange(6), yticklabels=BS, xlabel='Genesis month')
    title(a, 'b', 'Within-basin seasonality')
    cb = fig.add_axes([0.645, 0.55, 0.32, 0.012])
    fig.colorbar(im, cax=cb, orientation='horizontal', ticks=[0, 20, 40]).set_label('Cyclones within basin (%)', fontsize=8)
    a = ax[1, 0]
    a.stackplot(fraction.index, fraction.to_numpy().T, colors=[BC[k] for k in BS])
    a.set(xlabel='Genesis year', ylabel='Fraction of master sample', xlim=(1982, 2023), ylim=(0, 1), xticks=[1982, 1990, 2000, 2010, 2023])
    title(a, 'c', 'Basin composition')
    a.legend(handles=[Patch(color=BC[k], label=k) for k in BS], ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=7.5, columnspacing=1.1)
    a = ax[1, 1]
    values = support.to_numpy(float) / 2340 * 100
    support_image = a.imshow(values, aspect='auto', cmap='Greens', vmin=95, vmax=100)
    a.set(xticks=range(4), xticklabels=['D', 'E', 'F', 'Res'], yticks=range(6), yticklabels=VL, xlabel='Stage')
    title(a, 'd', 'Environmental support')
    for i in range(6):
        for j in range(4):
            a.text(j, i, str(int(support.iloc[i, j])), ha='center', va='center', fontsize=7.3, color='white' if values[i, j] > 98.3 else INK)
    support_cb = fig.add_axes([0.645, 0.063, 0.32, 0.013])
    support_cb.set_gid('d_colourbar')
    bar=fig.colorbar(support_image,cax=support_cb,orientation='horizontal',ticks=[95,97.5,100])
    bar.set_label('Available cyclones (%); cells: n',fontsize=7.5)
    bar.ax.tick_params(labelsize=7.5,length=2)
    a.set_xticks(np.arange(-0.5,4),minor=True);a.set_yticks(np.arange(-0.5,6),minor=True)
    a.grid(which='minor',color='white',linewidth=.5);a.tick_params(which='minor',length=0)
    export(fig, 'Coverage', exclude=[cb,support_cb])
