# Conf-Mar-2-Eri-NTU
Contains the all the files made for this conference

Suggested to use spyder to run the code. Any other environment with the necessary libraries imported will still be of no problem.

## Code Description for the file: Mar2_Code_With functions.py

#### Libraries to install
pyshp, scikit-learn, or-tools

#### Files to include in the code
Download the shaefile from this repo, fill the path in the place indicated below

     reader = shapefile.Reader(r"path of the file")
The first time you see the above code include the path of the file "AV Stops.shp" and the second time with the other shape file.

There is only one excel file. please include that in the code where you see teh below.

          all_points_df = pd.read_csv(r"path of the csv file")
          
#### Approach:
1. Considers the real loactions of defined AV_stops and parking locations
2. applies the heuristic developed for this research work and generates Vehicle-stop pairs
3. For each vehicle TSP is implemented where different algorithms can be tested
4. Results include Total vehicle travel miles.

#### Limitations:
1. TSP for different origin and destination is not optimal
2. The heuristic does not sort the demand values however it is not clear if this is a limitation


