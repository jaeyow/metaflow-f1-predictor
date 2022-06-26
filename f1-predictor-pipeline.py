from metaflow import FlowSpec, step
from datetime import datetime
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

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
		self.results_df.drop(columns=['Unnamed: 0'], inplace=True)
		print(self.results_df.head())
		print(self.results_df.shape)

		self.next(self.perform_feature_engineering)

	@step
	def perform_feature_engineering(self):
		"""
		Engineer features from the dataset
		"""
		import pandas as pd
		import numpy as np

		self.results_df['DriverExperience'] = 0

		drivers = self.results_df['Driver'].unique()
		for driver in drivers:
				df_driver = pd.DataFrame(self.results_df[self.results_df['Driver'] == driver]).tail(60)  # Arbitrary number, just look at the last x races
				df_driver.loc[:, 'DriverExperience'] = 1

				self.results_df.loc[self.results_df['Driver'] == driver, "DriverExperience"] = df_driver['DriverExperience'].cumsum()
				self.results_df['DriverExperience'].fillna(value=0, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Driver Experience

		***************************************************************************

			Driver's experience in Formula 1, where a more experienced F1 driver
			typically places better than a rookie.

			Added new feature: 'DriverExperience', new dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df['ConstructorExperience'] = 0
		constructors = self.results_df['Constructor'].unique()
		for constructor in constructors:
			df_constructor = pd.DataFrame(self.results_df[self.results_df['Constructor'] == constructor]).tail(60)  # Arbitrary number, just look at the last x races per driver
			df_constructor.loc[:, 'ConstructorExperience'] = 1

			self.results_df.loc[self.results_df['Constructor'] == constructor, "ConstructorExperience"] = df_constructor['ConstructorExperience'].cumsum()
			self.results_df['ConstructorExperience'].fillna(value=0, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Constructor Experience

		***************************************************************************

			Constructor's experience in Formula 1, where a more experienced F1
			constructor typically places better than a rookie.

			Added new feature: 'ConstructorExperience', new dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df['DriverRecentWins'] = 0
		drivers = self.results_df['Driver'].unique()

		self.results_df.loc[self.results_df['Position'] == 1, "DriverRecentWins"] = 1
		for driver in drivers:
				mask_first_place_drivers = (self.results_df['Driver'] == driver) & (self.results_df['Position'] == 1)
				df_driver = self.results_df[mask_first_place_drivers]
				self.results_df.loc[self.results_df['Driver'] == driver, "DriverRecentWins"] = self.results_df[self.results_df['Driver'] == driver]['DriverRecentWins'].rolling(60).sum()  # 60 races, about 3 years rolling
				# but don't count this race's win
				self.results_df.loc[mask_first_place_drivers, "DriverRecentWins"] = self.results_df[mask_first_place_drivers]['DriverRecentWins'] - 1
				self.results_df['DriverRecentWins'].fillna(value=0, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Driver Recent Wins

		***************************************************************************

			A new feature is added to represent the dirver's most recent past wins.
			Excluding the result of the current race ensures that there is no
			possibility of data leakage that might affect the results.

			Added new feature: 'DriverRecentWins', new dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df['DriverRecentDNFs'] = 0
		drivers = self.results_df['Driver'].unique()

		self.results_df.loc[(~self.results_df['Status'].str.contains(
				'Finished|\+')), "DriverRecentDNFs"] = 1
		for driver in drivers:
				mask_not_finish_place_drivers = (self.results_df['Driver'] == driver) & (~self.results_df['Status'].str.contains('Finished|\+'))
				df_driver = self.results_df[mask_not_finish_place_drivers]
				self.results_df.loc[self.results_df['Driver'] == driver, "DriverRecentDNFs"] = self.results_df[self.results_df['Driver']== driver]['DriverRecentDNFs'].rolling(60).sum()  # 60 races, about 3 years rolling
				self.results_df.loc[mask_not_finish_place_drivers, "DriverRecentDNFs"] = self.results_df[mask_not_finish_place_drivers]['DriverRecentDNFs'] - 1  # but don't count this race
				self.results_df['DriverRecentDNFs'].fillna(value=0, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Driver Recent DNFs

		***************************************************************************

			A new feature has also been added to represent a driver's recent DNFs
			(Did Not Finish), whatever/whoever's fault it is. We also have to take
			care and avoid data leakage into this new feature, by not counting
			the current race.

			Added new feature: 'DriverRecentDNFs', new dataframe shape: {self.results_df.shape}
		"""
		)

		print('Feature Engineering - Fix Recent Form Points')
		# Feature Engineering - Fix Recent Form Points
		# Add new RFPoints column - ALL finishers score points - max points First place and one less for each lesser place (using LogSpace)
		self.seasons = self.results_df['Season'].unique()
		self.results_df['RFPoints'] = 0
		for season in self.seasons:
				rounds = self.results_df[self.results_df['Season'] == season]['Round'].unique()
				for round in rounds:
						mask = (self.results_df['Season'] == season) & (self.results_df['Round'] == round)
						# Count only if finished the race
						finisher_mask = ((self.results_df['Status'].str.contains('Finished|\+')))
						finished_count = self.results_df.loc[(mask) & finisher_mask, "RFPoints"].count()
						# use list of LogSpaced numbers
						point_list = np.round(np.logspace(1, 4, 40, base=4), 4)
						point_list[::-1].sort()

						self.results_df.loc[(mask) & finisher_mask, "RFPoints"] = point_list[:finished_count].tolist()

		print(
		f"""
		***************************************************************************

			Feature Engineering - Recent Form Points

		***************************************************************************

			In Formula 1, only the top 10 finishers score points, so even if a driver
			finished 11th, they will not score anything which will not help our
			calculation. So in this part, we give all finishers a score. The 1st
			place top points, and lower places get lower points and so on. We can
			then use this column as a variable (instead of F1's official points)
			to calclulate for the the Driver's recent form.

			Added new feature: 'RFPoints', new dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df['DriverRecentForm'] = 0
		# for all drivers, calculate the rolling X DriverRecentForm and add to a new column in
		# original data frame, this represents the 'recent form', then for NA's just impute to zero
		drivers = self.results_df['Driver'].unique()
		for driver in drivers:
				df_driver = self.results_df[self.results_df['Driver'] == driver]
				self.results_df.loc[self.results_df['Driver'] == driver, "DriverRecentForm"] = df_driver['RFPoints'].rolling(30).sum() - df_driver['RFPoints']  # calcluate recent form points but don't include this race's points
				self.results_df['DriverRecentForm'].fillna(value=0, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Driver Recent Form

		***************************************************************************

			Now that we've got our adjusted points system "RFPoints", we can now
			calculate for a more accurate Driver Recent Form. We also have to take
			care and avoid data leakage into this new feature.

			Added new feature: 'DriverRecentForm', new dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df['ConstructorRecentForm'] = 0
		# for all constructors, calculate the rolling X RFPoints and add to a new column in
		# original data frame, this represents the 'recent form', then for NA's just impute to zero
		constructors = self.results_df['Constructor'].unique()
		for constructor in constructors:
				df_constructor = self.results_df[self.results_df['Constructor'] == constructor]
				self.results_df.loc[self.results_df['Constructor'] == constructor, "ConstructorRecentForm"] = df_constructor['RFPoints'].rolling(30).sum() - df_constructor['RFPoints']  # calcluate recent form points but don't include this race's points
				self.results_df['ConstructorRecentForm'].fillna(value=0, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Constructor Recent Form

		***************************************************************************

			Now that we've got our adjusted points system "RFPoints", we can now also
			calculate for a more accurate Constructor Recent Form. We also have to
			take care and avoid data leakage into this new feature.

			Added new feature: 'ConstructorRecentForm', new dataframe shape: {self.results_df.shape}
		"""
		)

		def calculate_age(born, race):
				date_born = datetime.strptime(born, '%Y-%m-%d')
				date_race = datetime.strptime(race, '%Y-%m-%d')
				return date_race.year - date_born.year - ((date_race.month, date_race.day) < (date_born.month, date_born.day))

		self.results_df['Age'] = self.results_df.apply(lambda x: calculate_age(x['DOB'], x['Race Date']), axis=1)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Driver Age

		***************************************************************************

			Surely a driver's age has some effect and may have some influence to
			the outcome of the race.

			Added new feature: 'Age', new dataframe shape: {self.results_df.shape}
		"""
		)

		def is_race_in_home_country(driver_nationality, race_country):
				nationality_country_map = {
						'American': ['USA'],
						'American-Italian': ['USA', 'Italy'],
						'Argentine': ['Argentina'],
						'Argentine-Italian': ['Argentina', 'Italy'],
						'Australian': ['Australia'],
						'Austrian': ['Austria'],
						'Belgian': ['Belgium'],
						'Brazilian': ['Brazil'],
						'British': ['UK'],
						'Canadian': ['Canada'],
						'Chilean': ['Brazil'],
						'Colombian': ['Brazil'],
						'Czech': ['Austria', 'Germany'],
						'Danish': ['Germany'],
						'Dutch': ['Netherlands'],
						'East German': ['Germany'],
						'Finnish': ['Germany', 'Austria'],
						'French': ['France'],
						'German': ['Germany'],
						'Hungarian': ['Hungary'],
						'Indian': ['India'],
						'Indonesian': ['Singapore', 'Malaysia'],
						'Irish': ['UK'],
						'Italian': ['Italy'],
						'Japanese': ['Japan', 'Korea'],
						'Liechtensteiner': ['Switzerland', 'Austria'],
						'Malaysian': ['Malaysia', 'Singapore'],
						'Mexican': ['Mexico'],
						'Monegasque': ['Monaco'],
						'New Zealander': ['Australia'],
						'Polish': ['Germany'],
						'Portuguese': ['Portugal'],
						'Rhodesian': ['South Africa'],
						'Russian': ['Russia'],
						'South African': ['South Africa'],
						'Spanish': ['Spain', 'Morocco'],
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
						print(
								"An exception occurred, This driver has no race held in his home country.")
				return race_country in countries

		self.results_df['IsHomeCountry'] = self.results_df.apply(lambda x: is_race_in_home_country(x['Nationality'], x['Country']), axis=1)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Home Circuit

		***************************************************************************

			Is there such a thing as Homecourt Advantage in Formula 1 racing? It doesn't
			look like it does, based on the preliminary EDA, however, I've got a feeling
			that it might have some. In the following cell, I have created a mapping
			between driver nationality vs race country, and this is used when we want
			to convey the Homecourt advantage concept in this model.

			Added new feature: 'IsHomeCountry', new dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df = pd.get_dummies(self.results_df, columns=['Weather', 'Nationality', 'Race Name'], drop_first=True)

		for col in self.results_df.columns:
				if 'Nationality' in col and self.results_df[col].sum() < 300:
						self.results_df.drop(col, axis=1, inplace=True)

				elif 'Race Name' in col and self.results_df[col].sum() < 130:
						self.results_df.drop(col, axis=1, inplace=True)

				else:
						pass

		print(
		f"""
		***************************************************************************

			Feature Engineering - Dummify categorical features

		***************************************************************************

			Dummify applicable categorical variables and ensure that the variables for the model are all numeric.

			- Race name (circuit)
			- Driver nationality

			New dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df.drop(['Race Date', 'Race Time', 'Status', 'DOB', 'Constructor', 'Constructor Nat', 'Circuit Name',
													'Race Url', 'Lat', 'Long', 'Locality', 'Country', 'Laps', 'Points',
													'RFPoints'], axis=1, inplace=True)

		print(
		f"""
		***************************************************************************

			Feature Engineering - Drop Columns which are not needed/required for modelling

		***************************************************************************

			Drop irrelevant columns to prevent model bloat and improve training and inference time

			New dataframe shape: {self.results_df.shape}
		"""
		)

		self.results_df['Season'] = pd.to_numeric(self.results_df['Season'])

		print(
		f"""
		***************************************************************************

			Feature Engineering - convert Season to numeric

		***************************************************************************

			Convert to numeric to help the model understand the racing season

			New dataframe shape: {self.results_df.shape}
		"""
		)

		# Prepare train set for the training
		# now setup the global variable to keep track of results
		self.scoring_raw ={'model':[], 'params': [], 'score': [], 'train_time': [], 'test_time': []}

		from sklearn.preprocessing import MinMaxScaler

		np.set_printoptions(precision=4)
		self.model_df = self.results_df.copy()
		self.model_df['Position'] = self.model_df['Position'].map(lambda x: 1 if x == 1 else 0)

		train = self.model_df[(self.model_df['Season'] >= 1950) & (self.model_df['Season'] < 2021)]
		X_train = train.drop(['Position','Driver'], axis = 1)
		self.y_train = train['Position']

		self.scaler = MinMaxScaler()
		self.X_train = pd.DataFrame(self.scaler.fit_transform(X_train, self.y_train), columns = X_train.columns)

		self.next(
			self.joses_dumb_classifier_train_and_test,
			self.linear_regression_train_and_test,
			self.adaboost_regression_train_and_test)

	def regression_test_score(self, model, print_output=False):
		from sklearn.metrics import precision_score
		
		# --- Test ---
		score = 0
		races = self.model_df[(self.model_df['Season'] == 2021)]['Round'].unique()
		for race in races:
				test = self.model_df[(self.model_df['Season'] == 2021) & (self.model_df['Round'] == race)]
				X_test = test.drop(['Position','Driver'], axis = 1)
				y_test = test['Position']
				X_test = pd.DataFrame(self.scaler.transform(X_test), columns = X_test.columns)
				X_test.to_csv(f'{2021}_{race}.csv')

				# make predictions
				prediction_df = pd.DataFrame(model.predict(X_test), columns = ['prediction'])
				merged_df = pd.concat([prediction_df, test[['Driver','Position']].reset_index(drop=True)], axis=1)
				merged_df.rename(columns = {'Position': 'actual_pos'}, inplace = True)
				
				# shuffle data to remove original order that will influence selection
				# of race winner when there are drivers with identical win probablilities
				merged_df = merged_df.sample(frac=1).reset_index(drop=True) 
				merged_df.sort_values(by='prediction', ascending=False, inplace=True)
				merged_df['predicted_pos'] = merged_df['prediction'].map(lambda x: 0)
				merged_df.iloc[0, merged_df.columns.get_loc('predicted_pos')] = 1
				merged_df.reset_index(drop=True, inplace=True)
				if (print_output == True):
						print(merged_df)

				# --- Score --- 
				score += precision_score(merged_df['actual_pos'], merged_df['predicted_pos'], zero_division=0)
				
		return score / len(races)

	@step
	def joses_dumb_classifier_train_and_test(self):
		"""
		Jose's Dumb Classifier
		"""
		import numpy as np
		from timeit import default_timer as timer

		class F1OracleDumbClassifier(BaseEstimator):
			def fit(self, X, y=None):
					pass
			def predict(self, X):
					# numbers between 0.001 to 0.99 and shuffle - jose's awesome dumb classifier!
					numbers = np.round(np.logspace(np.log10(0.001),np.log10(0.999),len(X)),4)
					np.random.shuffle(numbers)
					return numbers

		def joses_dumb_classifier(X_train, y_train):
			start = timer()
			model = F1OracleDumbClassifier()
			model.fit(X_train, y_train)
			end = timer()
			self.scoring_raw['train_time'].append(end - start)

			start = timer()
			model_score = self.regression_test_score(model)
			end = timer()

			self.scoring_raw['model'].append("Jose's Dumb Classifier")
			self.scoring_raw['score'].append(np.round(model_score*100, 3))
			self.scoring_raw['params'].append('none')
			self.scoring_raw['test_time'].append(end - start)

		print(
		f"""
		***************************************************************************

			Jose's Dumb Classifier

		***************************************************************************

		"""
		)
		joses_dumb_classifier(self.X_train, self.y_train)
		self.running_score = pd.DataFrame(self.scoring_raw)

		self.next(self.test_model_join)

	@step
	def linear_regression_train_and_test(self):
		"""
		Linear Regression
		"""
		from timeit import default_timer as timer
		from sklearn.linear_model import LinearRegression

		def linear_regression(X_train, y_train):
			params={'fit_intercept': ['True', 'False']}

			for fit_intercept in params['fit_intercept']:
				start = timer()
				model_params = (fit_intercept)
				model = LinearRegression(fit_intercept = fit_intercept)
				model.fit(X_train, y_train)
				end = timer()
				self.scoring_raw['train_time'].append(end - start)

				start = timer()
				model_score = self.regression_test_score(model)
				end = timer()

				self.scoring_raw['model'].append('Linear Regression')
				self.scoring_raw['score'].append(np.round(model_score*100, 3))
				self.scoring_raw['params'].append(model_params)
				self.scoring_raw['test_time'].append(end - start)
			
		print(
		f"""
		***************************************************************************

			Linear Regression

		***************************************************************************

		"""
		)

		linear_regression(self.X_train, self.y_train)
		self.running_score = pd.DataFrame(self.scoring_raw)	
		self.next(self.test_model_join)

	@step
	def adaboost_regression_train_and_test(self):
		"""
		AdaBoost Regression
		"""
		from timeit import default_timer as timer
		from sklearn.ensemble import AdaBoostRegressor

		def adaboost_regressor(X_train, y_train):
			params={'n_estimators': [100,200,300],
					'learning_rate': [0.001,0.01,0.1,1],
					'loss': ['linear','square','exponential']}

			for n_estimators in params['n_estimators']:
				for learning_rate in params['learning_rate']:
					for loss in params['loss']:
						start = timer()
						model_params = (n_estimators, learning_rate, loss)
						model = AdaBoostRegressor(random_state=0, n_estimators=n_estimators, learning_rate=learning_rate, loss=loss)
						model.fit(X_train, y_train)
						end = timer()
						self.scoring_raw['train_time'].append(end - start)

						start = timer()
						model_score = self.regression_test_score(model)
						end = timer()

						self.scoring_raw['model'].append('AdaBoost Regressor')
						self.scoring_raw['score'].append(np.round(model_score*100, 3))
						self.scoring_raw['params'].append(model_params)
						self.scoring_raw['test_time'].append(end - start)
			
		print(
		f"""
		***************************************************************************

			AdaBoost Regression

		***************************************************************************

		"""
		)

		adaboost_regressor(self.X_train, self.y_train)
		self.running_score = pd.DataFrame(self.scoring_raw)	
		self.next(self.test_model_join)

	@step
	def test_model_join(self, join_inputs):
			"""
			Join our parallel model training branches and decide the winning model
			"""
			print(f'F1PredictorPipeline ==> test_model_join...')
			# for input in join_inputs:
			# 		print(f'F1PredictorPipeline ==>  Using {input.algorithm}...')

			print(self.results_df.head(200))

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
