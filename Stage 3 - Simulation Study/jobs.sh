

# version 0.1.1+31.g1dde98a

##################################
### DATASET: meta_EA_final
##################################

# Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output/simulation
mkdir output/simulation/meta_EA_final/

# Collect descriptives about the dataset
mkdir output/simulation/meta_EA_final/descriptives
asreview stat data/meta_EA_final.csv -o output/simulation/meta_EA_final/descriptives/data_stats_meta_EA_final.json
asreview wordcloud data/meta_EA_final.csv -o output/simulation/meta_EA_final/descriptives/wordcloud_meta_EA_final.png --width 800 --height 500
asreview wordcloud data/meta_EA_final.csv -o output/simulation/meta_EA_final/descriptives/wordcloud_meta_EA_final_relevant.png --width 800 --height 500 --relevant
asreview wordcloud data/meta_EA_final.csv -o output/simulation/meta_EA_final/descriptives/wordcloud_meta_EA_final_irrelevant.png --width 800 --height 500 --irrelevant

# Simulate runs
mkdir output/simulation/meta_EA_final/state_files
asreview simulate data/meta_EA_final.csv -s output/simulation/meta_EA_final/state_files/state_meta_EA_final_65.h5 --prior_record_id 65 230 1059 36 996 73 127 693 783 291 1078 --seed 165
asreview simulate data/meta_EA_final.csv -s output/simulation/meta_EA_final/state_files/state_meta_EA_final_238.h5 --prior_record_id 238 230 1059 36 996 73 127 693 783 291 1078 --seed 165
asreview simulate data/meta_EA_final.csv -s output/simulation/meta_EA_final/state_files/state_meta_EA_final_775.h5 --prior_record_id 775 230 1059 36 996 73 127 693 783 291 1078 --seed 165
asreview simulate data/meta_EA_final.csv -s output/simulation/meta_EA_final/state_files/state_meta_EA_final_865.h5 --prior_record_id 865 230 1059 36 996 73 127 693 783 291 1078 --seed 165

# Collect metrics and plots about runs
asreview stat output/simulation/meta_EA_final/state_files/state_meta_EA_final_65.h5 output/simulation/meta_EA_final/state_files/state_meta_EA_final_238.h5 output/simulation/meta_EA_final/state_files/state_meta_EA_final_775.h5 output/simulation/meta_EA_final/state_files/state_meta_EA_final_865.h5  -o output/simulation/meta_EA_final/simulation_metrics_meta_EA_final.json
python scripts/get_atd.py -s output/simulation/meta_EA_final/state_files/ -d data/meta_EA_final.csv -o output/simulation/meta_EA_final/atd_meta_EA_final.csv
python scripts/merge_descriptives.py
python scripts/merge_metrics.py

# plots
asreview plot output/simulation/meta_EA_final/state_files/ output/simulation/meta_EA_final/state_files/state_meta_EA_final_65.h5 output/simulation/meta_EA_final/state_files/state_meta_EA_final_238.h5 output/simulation/meta_EA_final/state_files/state_meta_EA_final_775.h5 output/simulation/meta_EA_final/state_files/state_meta_EA_final_865.h5  --type inclusion -o output/simulation/meta_EA_final/plot_recall_meta_EA_final.png --show-absolute-values

##################################
### DATASET: meta_FT
##################################

# Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output/simulation
mkdir output/simulation/meta_FT/

# Collect descriptives about the dataset
mkdir output/simulation/meta_FT/descriptives
asreview stat data/meta_FT.csv -o output/simulation/meta_FT/descriptives/data_stats_meta_FT.json
asreview wordcloud data/meta_FT.csv -o output/simulation/meta_FT/descriptives/wordcloud_meta_FT.png --width 800 --height 500
asreview wordcloud data/meta_FT.csv -o output/simulation/meta_FT/descriptives/wordcloud_meta_FT_relevant.png --width 800 --height 500 --relevant
asreview wordcloud data/meta_FT.csv -o output/simulation/meta_FT/descriptives/wordcloud_meta_FT_irrelevant.png --width 800 --height 500 --irrelevant

# Simulate runs
mkdir output/simulation/meta_FT/state_files
asreview simulate data/meta_FT.csv -s output/simulation/meta_FT/state_files/state_meta_FT_65.h5 --prior_record_id 65 542 946 599 34 156 254 1326 1081 1074 567 --seed 166
asreview simulate data/meta_FT.csv -s output/simulation/meta_FT/state_files/state_meta_FT_238.h5 --prior_record_id 238 542 946 599 34 156 254 1326 1081 1074 567 --seed 166
asreview simulate data/meta_FT.csv -s output/simulation/meta_FT/state_files/state_meta_FT_775.h5 --prior_record_id 775 542 946 599 34 156 254 1326 1081 1074 567 --seed 166
asreview simulate data/meta_FT.csv -s output/simulation/meta_FT/state_files/state_meta_FT_865.h5 --prior_record_id 865 542 946 599 34 156 254 1326 1081 1074 567 --seed 166

# Collect metrics and plots about runs
asreview stat output/simulation/meta_FT/state_files/state_meta_FT_65.h5 output/simulation/meta_FT/state_files/state_meta_FT_238.h5 output/simulation/meta_FT/state_files/state_meta_FT_775.h5 output/simulation/meta_FT/state_files/state_meta_FT_865.h5  -o output/simulation/meta_FT/simulation_metrics_meta_FT.json
python scripts/get_atd.py -s output/simulation/meta_FT/state_files/ -d data/meta_FT.csv -o output/simulation/meta_FT/atd_meta_FT.csv
python scripts/merge_descriptives.py
python scripts/merge_metrics.py

# plots
asreview plot output/simulation/meta_FT/state_files/ output/simulation/meta_FT/state_files/state_meta_FT_65.h5 output/simulation/meta_FT/state_files/state_meta_FT_238.h5 output/simulation/meta_FT/state_files/state_meta_FT_775.h5 output/simulation/meta_FT/state_files/state_meta_FT_865.h5  --type inclusion -o output/simulation/meta_FT/plot_recall_meta_FT.png --show-absolute-values

##################################
### DATASET: meta_original
##################################

# Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output/simulation
mkdir output/simulation/meta_original/

# Collect descriptives about the dataset
mkdir output/simulation/meta_original/descriptives
asreview stat data/meta_original.csv -o output/simulation/meta_original/descriptives/data_stats_meta_original.json
asreview wordcloud data/meta_original.csv -o output/simulation/meta_original/descriptives/wordcloud_meta_original.png --width 800 --height 500
asreview wordcloud data/meta_original.csv -o output/simulation/meta_original/descriptives/wordcloud_meta_original_relevant.png --width 800 --height 500 --relevant
asreview wordcloud data/meta_original.csv -o output/simulation/meta_original/descriptives/wordcloud_meta_original_irrelevant.png --width 800 --height 500 --irrelevant

# Simulate runs
mkdir output/simulation/meta_original/state_files
asreview simulate data/meta_original.csv -s output/simulation/meta_original/state_files/state_meta_original_65.h5 --prior_record_id 65 1136 657 1316 269 1255 668 666 1135 672 460 --seed 167
asreview simulate data/meta_original.csv -s output/simulation/meta_original/state_files/state_meta_original_238.h5 --prior_record_id 238 1136 657 1316 269 1255 668 666 1135 672 460 --seed 167
asreview simulate data/meta_original.csv -s output/simulation/meta_original/state_files/state_meta_original_775.h5 --prior_record_id 775 1136 657 1316 269 1255 668 666 1135 672 460 --seed 167
asreview simulate data/meta_original.csv -s output/simulation/meta_original/state_files/state_meta_original_865.h5 --prior_record_id 865 1136 657 1316 269 1255 668 666 1135 672 460 --seed 167

# Collect metrics and plots about runs
asreview stat output/simulation/meta_original/state_files/state_meta_original_65.h5 output/simulation/meta_original/state_files/state_meta_original_238.h5 output/simulation/meta_original/state_files/state_meta_original_775.h5 output/simulation/meta_original/state_files/state_meta_original_865.h5  -o output/simulation/meta_original/simulation_metrics_meta_original.json
python scripts/get_atd.py -s output/simulation/meta_original/state_files/ -d data/meta_original.csv -o output/simulation/meta_original/atd_meta_original.csv
python scripts/merge_descriptives.py
python scripts/merge_metrics.py

# plots
asreview plot output/simulation/meta_original/state_files/ output/simulation/meta_original/state_files/state_meta_original_65.h5 output/simulation/meta_original/state_files/state_meta_original_238.h5 output/simulation/meta_original/state_files/state_meta_original_775.h5 output/simulation/meta_original/state_files/state_meta_original_865.h5  --type inclusion -o output/simulation/meta_original/plot_recall_meta_original.png --show-absolute-values

