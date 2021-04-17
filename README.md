# cell-analyzer
A Python library for segmenting and analyzing cells


# how to download cell-analyzer
Do either of the following:
	A. Download the repo zip file and unpack in desired location
or
	B. Navigate to desired location and run git clone:
		git clone https://github.com/esorelle/cell-analyzer


# info before using cell-analyzer
• cell-analyzer was developed to study whole-cell fluorescence of EBV+ cells
• cell-analyzer is designed to analyze TIFF image files (extension '.tif' or '.TIF')
• for now, out-of-the-box, image files must conform to the following format:

	YYMMDD_XXXXXX_L##_##_d##_img###.tif
	• YYMMDD is a date identifier
	• XXXXXX can by any 6 characters to identify cell line, etc.
	• L## can be any letter and two numbers; ex: A01 (for well A01 on a plate)
	• d## can specify the experimental timepoint; ex: d07 (day 7 post-event)
	• img### specifies the image replicate for a single condition; ex: img001

• for now, cell-analyzer only supports single-channel grayscale TIFF iamges
• feel free to make modifications based on your file names, data format, etc.
• these aspects are currently under development


# how to use cell-analyzer
• cell-analyzer can be run using python3 from the command line:
	>>> python3 ./path/to/cell-analyzer/run.py './path/to/image_directory'
• the target image directory is the only required argument
• optional arguments:
	--segment		[default is True, performs image segmentation]
	--analyze		[default is True, performs clustering analysis]
	--max_cell_area		[currently not used -- under development]
	--small_obj_size	[default is 300, sets min cell size in pixels]
	--num_clust		[default is 7, specifies # of clusters to ID]


# how to modify or customize cell-analyzer for your project
• see 'run.py' for more detail on each argument and function calls
• see 'cell_analyzer/analyzer_core.py' for process, file i/o, and analysis


# refer to '__TO-DO_LIST__.txt' for planned updates on dev branch


There's a lot of great cell segmentation software out there.
This is a very small project, mostly for fun.
If you find it useful or make cool improvements to the code for your work...
...then please remember to cite this project. Thanks!
