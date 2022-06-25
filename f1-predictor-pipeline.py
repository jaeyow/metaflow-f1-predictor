from metaflow import FlowSpec, step
from datetime import datetime

class F1PredictorPipeline(FlowSpec):
  """
  F1PredictorPipeline is an end-to-end flow for F1 Predictor
  """

  @step
  def start(self):
    """
    Initialization, place everything init related here, check that everything is
    in order like environment variables, connection strings, etc, and if there are
    any issues, fail fast here, now.
    """
    print(f'F1PredictorPipeline ==> start...')
    self.next(self.transform_data)

  @step
  def transform_data(self):
    """
    Collect data from Ergast API, transform and push to datawarehouse
    """
    print(f'F1PredictorPipeline ==> transform_data...')
    self.next(self.get_dataset)

  @step
  def get_dataset(self):
    """
    Retrieve data from datawarehouse
    """
    import awswrangler as wr

    print(f'F1PredictorPipeline ==> get_dataset...')
    self.results_df = wr.s3.read_csv('s3://metaflow-f1-predictor/part-1/csvs/latest_race_results.csv')
    self.results_df.drop(columns=['Unnamed: 0'],inplace=True)
    print(self.results_df.head())
    print(self.results_df.columns)

    self.next(self.engineer_features)

  # def calculate_age(self, born, race):
  #     date_born = datetime.strptime(born,'%Y-%m-%d')
  #     date_race = datetime.strptime(race,'%Y-%m-%d')
  #     return date_race.year - date_born.year - ((date_race.month, date_race.day) < (date_born.month, date_born.day))

  @step
  def engineer_features(self):
    """
    Engineer features from the dataset
    """
    import pandas as pd
    import numpy as np

    print(f'F1PredictorPipeline ==> engineer_features...')

    print('Feature Engineering - Driver Experience')
    self.results_df['DriverExperience'] = 0

    self.drivers = self.results_df['Driver'].unique()
    for driver in self.drivers:
        df_driver = pd.DataFrame(self.results_df[self.results_df['Driver']==driver]).tail(60) # Arbitrary number, just look at the last x races
        df_driver.loc[:,'DriverExperience'] = 1
        
        self.results_df.loc[self.results_df['Driver']==driver, "DriverExperience"] = df_driver['DriverExperience'].cumsum()
        self.results_df['DriverExperience'].fillna(value=0,inplace=True)

    print('Feature Engineering - Constructor Experience')
    # Feature Engineering - Constructor Experience
    self.results_df['ConstructorExperience'] = 0
    self.constructors = self.results_df['Constructor'].unique()
    for constructor in self.constructors:
        df_constructor = pd.DataFrame(self.results_df[self.results_df['Constructor']==constructor]).tail(60) # Arbitrary number, just look at the last x races per driver
        df_constructor.loc[:,'ConstructorExperience'] = 1
        
        self.results_df.loc[self.results_df['Constructor']==constructor, "ConstructorExperience"] = df_constructor['ConstructorExperience'].cumsum()
        self.results_df['ConstructorExperience'].fillna(value=0,inplace=True)

    print('Feature Engineering - Driver Recent Wins')
    # Feature Engineering - Driver Recent Wins
    self.results_df['DriverRecentWins'] = 0
    self.drivers = self.results_df['Driver'].unique()

    self.results_df.loc[self.results_df['Position']==1, "DriverRecentWins"] = 1
    for driver in self.drivers:
        mask_first_place_drivers = (self.results_df['Driver']==driver) & (self.results_df['Position']==1)
        df_driver = self.results_df[mask_first_place_drivers]
        self.results_df.loc[self.results_df['Driver']==driver, "DriverRecentWins"] = self.results_df[self.results_df['Driver']==driver]['DriverRecentWins'].rolling(60).sum() # 60 races, about 3 years rolling
        self.results_df.loc[mask_first_place_drivers, "DriverRecentWins"] = self.results_df[mask_first_place_drivers]['DriverRecentWins'] - 1  # but don't count this race's win
        self.results_df['DriverRecentWins'].fillna(value=0,inplace=True)

    print('Feature Engineering - Driver Recent DNFs')
    # Feature Engineering - Driver Recent DNFs
    self.results_df['DriverRecentDNFs'] = 0
    self.drivers = self.results_df['Driver'].unique()

    self.results_df.loc[(~self.results_df['Status'].str.contains('Finished|\+')), "DriverRecentDNFs"] = 1
    for driver in self.drivers:
        mask_not_finish_place_drivers = (self.results_df['Driver']==driver) & (~self.results_df['Status'].str.contains('Finished|\+'))
        df_driver = self.results_df[mask_not_finish_place_drivers]
        self.results_df.loc[self.results_df['Driver']==driver, "DriverRecentDNFs"] = self.results_df[self.results_df['Driver']==driver]['DriverRecentDNFs'].rolling(60).sum() # 60 races, about 3 years rolling
        self.results_df.loc[mask_not_finish_place_drivers, "DriverRecentDNFs"] = self.results_df[mask_not_finish_place_drivers]['DriverRecentDNFs'] - 1  # but don't count this race
        self.results_df['DriverRecentDNFs'].fillna(value=0,inplace=True)

    print('Feature Engineering - Fix Recent Form Points')
    # Feature Engineering - Fix Recent Form Points
    # Add new RFPoints column - ALL finishers score points - max points First place and one less for each lesser place (using LogSpace)
    self.seasons = self.results_df['Season'].unique()
    self.results_df['RFPoints'] = 0
    for season in self.seasons:
        rounds = self.results_df[self.results_df['Season']==season]['Round'].unique()
        for round in rounds:
            mask = (self.results_df['Season']==season) & (self.results_df['Round']==round)
            finisher_mask = ((self.results_df['Status'].str.contains('Finished|\+'))) # Count only if finished the race
            finished_count = self.results_df.loc[(mask) & finisher_mask, "RFPoints"].count()
            point_list = np.round(np.logspace(1,4,40, base=4),4) # use list of LogSpaced numbers
            point_list[::-1].sort()
            
            self.results_df.loc[(mask) & finisher_mask, "RFPoints"] = point_list[:finished_count].tolist()

    print('Feature Engineering - Driver Age')
    # Feature Engineering - Driver Age
    def calculate_age(born, race):
        date_born = datetime.strptime(born,'%Y-%m-%d')
        date_race = datetime.strptime(race,'%Y-%m-%d')
        return date_race.year - date_born.year - ((date_race.month, date_race.day) < (date_born.month, date_born.day))

    self.results_df['Age'] = self.results_df.apply(lambda x: calculate_age(x['DOB'], x['Race Date']), axis=1)

    print('Feature Engineering - Home Circuit')
    # Feature Engineering - Home Circuit
    def is_race_in_home_country(driver_nationality, race_country):
        nationality_country_map = {
            'American': ['USA'],
            'American-Italian': ['USA','Italy'],
            'Argentine': ['Argentina'],
            'Argentine-Italian': ['Argentina','Italy'],
            'Australian': ['Australia'],
            'Austrian': ['Austria'],
            'Belgian': ['Belgium'],
            'Brazilian': ['Brazil'],
            'British': ['UK'],
            'Canadian': ['Canada'],
            'Chilean': ['Brazil'],
            'Colombian': ['Brazil'],
            'Czech': ['Austria','Germany'],
            'Danish': ['Germany'],
            'Dutch': ['Netherlands'],
            'East German': ['Germany'],
            'Finnish': ['Germany','Austria'],
            'French': ['France'],
            'German': ['Germany'],
            'Hungarian': ['Hungary'],
            'Indian': ['India'],
            'Indonesian': ['Singapore','Malaysia'],
            'Irish': ['UK'],
            'Italian': ['Italy'],
            'Japanese': ['Japan','Korea'],
            'Liechtensteiner': ['Switzerland','Austria'],
            'Malaysian': ['Malaysia','Singapore'],
            'Mexican': ['Mexico'],
            'Monegasque': ['Monaco'],
            'New Zealander': ['Australia'],
            'Polish': ['Germany'],
            'Portuguese': ['Portugal'],
            'Rhodesian': ['South Africa'],
            'Russian': ['Russia'],
            'South African': ['South Africa'],
            'Spanish': ['Spain','Morocco'],
            'Swedish': ['Sweden'],
            'Swiss': ['Switzerland'],
            'Thai': ['Malaysia'],
            'Uruguayan': ['Argentina'],
            'Venezuelan': ['Brazil']
        }
        
        countries = ['None']
        
        try:
          countries = nationality_country_map[driver_nationality]
        except:
          print("An exception occurred, This driver has no race held in his home country.")
        return race_country in countries

    self.results_df['IsHomeCountry'] = self.results_df.apply(lambda x: is_race_in_home_country(x['Nationality'], x['Country']), axis=1)     

    print('Feature Engineering - Driver Recent Form')
    # Feature Engineering - Driver Recent Form
    self.results_df['DriverRecentForm'] = 0
    # for all drivers, calculate the rolling X DriverRecentForm and add to a new column in 
    # original data frame, this represents the 'recent form', then for NA's just impute to zero
    self.drivers = self.results_df['Driver'].unique()
    for driver in self.drivers:
        self.results_df = self.results_df[self.results_df['Driver']==driver]
        self.results_df.loc[self.results_df['Driver']==driver, "DriverRecentForm"] = self.results_df['RFPoints'].rolling(30).sum() - self.results_df['RFPoints'] # calcluate recent form points but don't include this race's points
        self.results_df['DriverRecentForm'].fillna(value=0,inplace=True)

    print('Feature Engineering - Constructor Recent Form')
    # Feature Engineering - Constructor Recent Form
    self.results_df['ConstructorRecentForm'] = 0
    # for all constructors, calculate the rolling X RFPoints and add to a new column in 
    # original data frame, this represents the 'recent form', then for NA's just impute to zero
    self.constructors = self.results_df['Constructor'].unique()
    for constructor in self.constructors:
        df_constructor = self.results_df[self.results_df['Constructor']==constructor]
        self.results_df.loc[self.results_df['Constructor']==constructor, "ConstructorRecentForm"] = df_constructor['RFPoints'].rolling(30).sum() - df_constructor['RFPoints'] # calcluate recent form points but don't include this race's points
        self.results_df['ConstructorRecentForm'].fillna(value=0,inplace=True)

    print('Feature Engineering - Dummify categorical features')
    # Feature Engineering - Dummify categorical features
    self.results_df = pd.get_dummies(self.results_df, columns = ['Weather', 'Nationality', 'Race Name'],drop_first=True)

    for col in self.results_df.columns:
        if 'Nationality' in col and self.results_df[col].sum() < 300:
            self.results_df.drop(col, axis = 1, inplace = True)
            
        elif 'Race Name' in col and self.results_df[col].sum() < 130:
            self.results_df.drop(col, axis = 1, inplace = True)
            
        else:
            pass

    print('Feature Engineering - Drop Columns which are not needed/required for modelling')
    # Feature Engineering - Drop Columns which are not needed/required for modelling
    self.results_df.drop(['Race Date', 'Race Time', 'Status', 'DOB', 'Constructor', 'Constructor Nat', 'Circuit Name',
                    'Race Url', 'Lat', 'Long', 'Locality', 'Country','Laps','Points',
                    'RFPoints'], axis=1, inplace=True)

    print('Feature Engineering - convert Season to numeric')
    # Feature Engineering - convert Season to numeric
    self.results_df['Season'] = pd.to_numeric(self.results_df['Season'])
    # print(f'Final: {self.results_df.columns}')
    print(self.results_df.head())

    self.algos = ["Algorithm 1", "Algorothm 2"]
    self.next(self.train_model, foreach="algos")

  @step
  def train_model(self):
    """
    Train Models in Parallel
    """
    print(f'F1PredictorPipeline ==> train_model...')
    self.algorithm = self.input
    self.next(self.test_model)

  @step
  def test_model(self):
    """
    Train Models in Parallel
    """
    print(f'F1PredictorPipeline ==> test_model...')
    self.next(self.test_model_join)

  @step
  def test_model_join(self, join_inputs):
    """
    Join our parallel model training branches and decide the winning model
    """
    print(f'F1PredictorPipeline ==> test_model_join...')
    for input in join_inputs:
      print(f'F1PredictorPipeline ==>  Using {input.algorithm}...')

    self.next(self.deploy_winning_model)

  @step
  def deploy_winning_model(self):
    """
    Deploy winning model to public API
    """
    print(f'F1PredictorPipeline ==> deploy_winning_model...')
    self.next(self.end)

  @step
  def end(self):
    """
    Anything that you want to do before finishing the pipeline is done here
    """
    print("F1PredictorPipeline is all done.")


if __name__ == "__main__":
    F1PredictorPipeline()
