from metaflow import FlowSpec, step, current
from comet_ml import API, Experiment
import os
import random

try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    print("No dotenv package")

class MyPipeline(FlowSpec):
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

        assert os.environ['COMET_API_KEY']

        comet_exp = Experiment(
          api_key=os.environ['COMET_API_KEY'],
          project_name="my-project",
          workspace="jaeyow",
          )

        self.jose = 'reyes'
        self.comet_experiment_key = comet_exp.get_key() # this is a string so it's safe to pickle

        self.next(
          self.train_model1,
          self.train_model2,
          self.train_model3,
          self.train_model4)

    @step
    def train_model1(self):
        """
        Train Models in Parallel
        """
        self.step_name = current.step_name
        comet_exp = API().get_experiment_by_key(self.comet_experiment_key)
        comet_exp.log_metric(f'Alg1_train_time', random.randint(1, 99))

        model_exp = Experiment(
          api_key=os.environ['COMET_API_KEY'],
          project_name="my-project",
          workspace="jaeyow",
          )

        model_exp.log_metric(f'Alg1_param1_1', random.randint(1, 99))
        model_exp.log_metric(f'Alg1_param1_2', random.randint(1, 99))
        model_exp.log_metric(f'Alg1_param1_3', random.randint(1, 99))
        model_exp.log_metric(f'Alg1_param1_4', random.randint(1, 99))

        print(f'Jose: {self.jose}')
        self.next(self.test_model_join)

    @step
    def train_model2(self):
        """
        Train Models in Parallel
        """
        comet_exp = API().get_experiment_by_key(self.comet_experiment_key)
        comet_exp.log_metric(f'Alg2_train_time', random.randint(1, 99))

        model_exp = Experiment(
          api_key=os.environ['COMET_API_KEY'],
          project_name="my-project",
          workspace="jaeyow",
          )

        model_exp.log_metric(f'Alg2_param1_1', random.randint(1, 99))
        model_exp.log_metric(f'Alg2_param1_2', random.randint(1, 99))
        model_exp.log_metric(f'Alg2_param1_3', random.randint(1, 99))
        model_exp.log_metric(f'Alg2_param1_4', random.randint(1, 99))

        self.step_name = current.step_name
        self.next(self.test_model_join)

    @step
    def train_model3(self):
        """
        Train Models in Parallel
        """
        comet_exp = API().get_experiment_by_key(self.comet_experiment_key)
        comet_exp.log_metric(f'Alg3_train_time', random.randint(1, 99))

        model_exp = Experiment(
          api_key=os.environ['COMET_API_KEY'],
          project_name="my-project",
          workspace="jaeyow",
          )

        model_exp.log_metric(f'Alg3_param1_1', random.randint(1, 99))
        model_exp.log_metric(f'Alg3_param1_2', random.randint(1, 99))
        model_exp.log_metric(f'Alg3_param1_3', random.randint(1, 99))
        model_exp.log_metric(f'Alg3_param1_4', random.randint(1, 99))

        self.step_name = current.step_name
        self.next(self.test_model_join)

    @step
    def train_model4(self):
        """
        Train Models in Parallel
        """
        comet_exp = API().get_experiment_by_key(self.comet_experiment_key)
        comet_exp.log_metric(f'Alg4_train_time', random.randint(1, 99))

        model_exp = Experiment(
          api_key=os.environ['COMET_API_KEY'],
          project_name="my-project",
          workspace="jaeyow",
          )

        model_exp.log_metric(f'Alg4_param1_1', random.randint(1, 99))
        model_exp.log_metric(f'Alg4_param1_2', random.randint(1, 99))
        model_exp.log_metric(f'Alg4_param1_3', random.randint(1, 99))
        model_exp.log_metric(f'Alg4_param1_4', random.randint(1, 99))

        self.step_name = current.step_name
        self.next(self.test_model_join)            

    @step
    def test_model_join(self, join_inputs):
        """
        Join our parallel model training branches and decide the winning model
        """
        # comet_exp = API().get_experiment_by_key(self.comet_experiment_key)

        # for input in join_inputs:
        #   print(f'Hey {input.alg} => {input.train_time} => {self.jose}...')
          # comet_exp.log_metric(f'{input.alg}_train_time', input.step_name, input.train_time)

        self.next(self.deploy_winning_model)    

    @step
    def deploy_winning_model(self):
        """
        Deploy winning model to public API
        """
        self.next(self.end)

    @step   
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """

if __name__ == "__main__":
    MyPipeline()