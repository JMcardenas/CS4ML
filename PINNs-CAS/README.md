# Figure 3 and Appendix D
Structure:
	- colab_data
	- colab_codes
	- colab_results
	
	- data
	- figs
	- src
	- results
	- utils
#-----------------------------------------------------------------------------------------------------#
Step 0: "Generate or load the data from MATLAB"
	- You can skip this step and move to step 1. Otherwise, do the following: 
		- run "generate_data.m" from folder "src/gen data/".
		- It generates the PDE data and saves it on "colab_data/".
		- compress the file as "colab_data.zip" or use the preload folder. 
		
Step 1: "Create the Google Colab notebook"
	- Use upload option to upload:
		- "CS4ML_Appendix_D.ipynb" from "colab_codes/" folder. 
		- colab_data.zip
		- colab_codes.zip
	
Step 2: "Run main code"
	- Hit run. This will execute three things:
		- Train all the DNNs in figure 3 and Appendix D.
		- Extract the data for all trials.
		- Save the folder "colab_results/" as "colab_results.zip".

Step 3: "Move data to local machine"
	- Download the file "colab_results.zip" and replace it by "colab_results/".
	- Notice that "colab_result/" has already the data. 
	
Step 4: "Extract data from several files"
	- Open MATLAB and run "extract_data.m" in "colab_codes/" folder. 
	- It will create the main data from "colab_results/" and save it in "results/" folder.
	
Step 5: "Generate data to plot"
	- Open MATLAB and run:
		- "fig_3_run.m" in src/ folder.
		- "fig_App_D_run.m" in src/ folder.
		
Step 6: "Plot"
	- Open MATLAB and run "fig_3_plot.m" and "fig_App_D_plot.m" in src/ folder. 
	- It will generate the plots for figure 3 and Appendix D.
	- The figures will be save in the folder figs/.
	 
	
