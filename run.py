#!/usr/bin/env python3

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
    help="""sets threshold (# pixels) above which regions are recursively split""")
@click.option(
    '--analyze', type=bool, default=True, show_default=True,
    help="""sets threshold (# pixels) above which regions are recursively split""")
@click.option(
    '--max_cell_area', type=int, default=20000, show_default=True,
    help="""sets threshold (# pixels) above which regions are recursively split""")
@click.option(
    '--small_obj_size', type=int, default=300, show_default=True,
    help="""sets minimum region size (# of pixels)""")
@click.option(
    '--num_clust', type=int, default=7, show_default=True,
    help="""sets # of clusters for segmented cell analysis""")
@click.argument('targetdirectory', type=click.Path(exists=True))  # no help statements for required arguments

def cli(
        segment,
        analyze,
        max_cell_area,
        small_obj_size,
        num_clust,
        targetdirectory
):
    if segment==True:
        print('segmenting cells in: ' + targetdirectory + '...')
        results_df, save_path = analyzer_core.read_and_process_directory(targetdirectory, max_cell_area, small_obj_size)
        print('...cell segmentation finished' + '\n')
    else:
        print('### no segmentation performed ###')

    if analyze==True:
        print('analzying segmented cells...')
        analyzer_core.cluster_results(results_df, save_path, num_clust)
        print('...analysis finished' + '\n')
    else:
        print('### no analysis performed ###')

if __name__ == '__main__':
    cli()
