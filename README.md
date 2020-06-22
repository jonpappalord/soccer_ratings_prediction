# soccer_ratings_prediction

The project consist in 7 different folders:
1. **marks_crawling_and_id_matching**: in this folder there are all the script to crawl data from Fantacalcio and Pianeta fantacalcio (Marks_Scraping_From_Existing_Csv and Marks_Scraping_From_Fantacalcio notebooks). Since the player from the dataset retrieved has a plain text name, there are also script that allow to pair the plain text name of the player to wyscout id (Table_Match_Creation for retrieve wyscoutplayer ids and Merge_Different_Csv_Sources that use some string comparison to make the pair). 
    1. **marks_scraping_from_existing_csv**: Regarding input, in the data folder another folder is needed called fantavoti_G-CDS-TS-201X201Y (where X and Y are the reference seasons). Inside this folder is needed a file 201X-201Y-other.csv (took from https://www.pianetafanta.it/Voti-Ufficiali-Archivio.asp). This task give as output a dataset called alltheotherScore_201X_201Y.csv
    2. **marks_scraping_from_Fantacalcio**: take no dataset as input. Give as output a dataset called fantacalcioScore_201X_201Y.csv
    3. **table_matching_creation**: take as input the wyscout datasets players.json and teams.json. Return a csv called wy_scout_player_association_name_id.csv.
    4. **merge_different_csv_source**: take as input the dataset obtained from point 1.1, 1.2 and 1.3. So starting from alltheotherScore_201X_201Y.csv, fantacalcioScore_201X_201Y.csv and wy_scout_player_association_name_id.csv give as result a dataset called final_df_season_201X_201Y.csv

2. **feature_extraction**: in this folder there is a single notebook that starting from a dataset composed by match id, player id and gameweek, using the wyscout events dataset create a dataset that pass through all the features write in the thesis.
    * **features_extraction_from_wyscout**: take as input wyscout datasets player.json, teams.json, matches.json (of the season we want to extract the features) and events.json (again of the season we want to extract the features). As output give a file called dataset_player_match_season_201X_201Y_featuresextracted.csv.

3. **join_dataset_created_for_the_model**: merge the dataset of marks, with the dataset of features. Moreover add the ratings computed from player rank application https://github.com/mesosbrodleto/playerank.
    * **dataset_for_model_creation**: Take as input the datasets obtained from 1.4 and 2. (respectively final_df_season_201X_201Y.csv and dataset_player_match_season_201X_201Y_featuresextracted.csv). Then using the notebook provided by playerrank we used also the player_rank_score_201X_201Y.csv dataset. The output of the process give a dataset called for_the_model_201X_201Y.csv.

4. **data_analysis**: explore the dataset obtained from the whole processing. The key concept is to understand marks distribution and relations; moreover it consider some correlation bewteen computed features w.r.t. gazzetta marks and fantacalcio marks. At the end is provided also a reason over the evolution of players during a season analyzing their path. plot_analysis.py contains all the function used in the nootebook data_analysis_with_explanations.
    * **data_analysis_with_explanations**: take as input a dataset from a single season for_the_model_201X_201Y.csv

5. **model_creation_and_evaluation**: here I applyied different well known machine learning algorithm in order to found out the model that better fit with the problem of ordinal prediction. In details I explored an ordinal linear regressor (from mord package), 2 different neural networks applications, an XGBoost application and a decision tree regressor. For each different algorithm I tuned all the parameters and at the end compare the different outcome. At the end of the whole process came out that mord ordinal linear regressor overcome the other ones.
    * **model_creation_and_evaluation**: based on the user preferences requires dataset of a whole season, it requires at least 1 season. The format of the names of the input data is of this structure for_the_model_201X_201Y.csv

6. **predictions_explanation**: In the very first part of the notebook I divided the whole dataset obtained in training and test set (in particular the test set is composed by 10 different matches of season 2018/2019). Then using the best algorithm obtained from the previous notebook I trained a model for each role. At the end using different framework I computed global feature importance and local feature importance. In details regarding Global feature importance I used Skater framework and plot the metrics that influence the most the whole predictions in a radar chart; instead for local predictions using Shap framework I took 4 examples (one for each role) and plot the motivations that move the predictor to assign a determined probability to that local case. **Recall that the shap function to compute shap value is slow, so in order to increase the efficency we put 4 .txt files (one for each role) that contains already computed shap values for season 2016-2017, 2017-2018 and 2018-2019**, to access these files put, as first parameter in function locals_explanation_using_shap, the attribute 'load'.
    * **predictions_explanations**: Take as input all the seasons the user want to use (name in format for_the_model_201X_201Y.csv) and, since we need to make a real experiment, this notebook also required matches.json from season 2018-2019

7. **experiment_csv_creation**: Combining the result of the predictions with the local motivations obtained from shap, I setup an experiment in which, for each of the 10 game selected as test set a comparison, some partecipant will give their opinion to each player that played the game in order to retrieve a comparision between human perception of perfomance and machine perception of performance. 