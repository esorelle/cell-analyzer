#! /usr/bin/python3

import click
from cell_analyzer import analyzer_core


@click.command(
    'find-cells',
    help="""analyzes images to segment and analyze cells
    from the command line. The image data path (targetdirectory) must
    exist and be passed as an argument. See below for optional arguments
    and defaults.""")
@click.option(
    '--segment', type=bool, default=True, show_default=True,
    help="""conditional to segment images in target directory""")
@click.option(
    '--analyze', type=bool, default=True, show_default=True,
    help="""conditional to perform unsupervised hierarchical clustering of cell data""")
@click.option(
    '--norm_window', type=int, default=401, show_default=True,
    help="""sets size of local adaptive normalization window""")
@click.option(
    '--min_hole_size', type=int, default=50000, show_default=True,
    help="""sets size of cell holes to fill""")
@click.option(
    '--min_cell_size', type=int, default=1000, show_default=True,
    help="""sets minimum cell area (# of pixels)""")
@click.option(
    '--extrema_blur', type=int, default=15, show_default=True,
    help="""sets gaussian blur sigma for coarse local extrema for watershed""")
@click.option(
    '--peak_sep', type=int, default=15, show_default=True,
    help="""sets minimum distance between local extrema for watershed""")
@click.option(
    '--formatted_titles', type=bool, default=False, show_default=True,
    help="""conditional addition of formatted title metadata to results dataframes""")
@click.option(
    '--channel_list', type=list, default=[], show_default=True,
    help="""specify image channel string (e.g., --channel_list = '12' for channels 1 and 2) to perform intra-cell feature segmentation"""
)
@click.option(
    '--num_clust', type=int, default=10, show_default=True,
    help="""sets # of clusters for segmented cell analysis""")
@click.argument('targetdirectory', type=click.Path(exists=True))  # no help statements for required arguments

def cli(
        segment,
        analyze,
        norm_window,
        min_hole_size,
        min_cell_size,
        extrema_blur,
        peak_sep,
        formatted_titles,
        channel_list,
        num_clust,
        targetdirectory
):
    if segment==True:
        print('segmenting cells in: ' + targetdirectory + '...')
        results_df, save_path, n_chan = analyzer_core.read_and_process_directory(targetdirectory, norm_window, min_hole_size, min_cell_size, extrema_blur, peak_sep, formatted_titles, channel_list)
        print('...cell segmentation finished' + '\n')
    else:
        print('### no segmentation performed ###')

    if analyze==True:
        print('analzying segmented cells...')
        analyzer_core.cluster_results(results_df, save_path, n_chan, num_clust, formatted_titles, channel_list)
        print('...analysis finished' + '\n')
    else:
        print('### no analysis performed ###')

if __name__ == '__main__':
    cli()
