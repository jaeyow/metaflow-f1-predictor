from metaflow import FlowSpec, step

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
        Collect data from Ergast, transform and push to datawarehouse
        """
        print(f'F1PredictorPipeline ==> transform_data...')
        self.next(self.get_dataset)

    @step
    def get_dataset(self):
        """
        Retrieve data from datawarehouse
        """
        print(f'F1PredictorPipeline ==> get_dataset...')
        self.next(self.engineer_features)

    @step
    def engineer_features(self):
        """
        Engineer features from the dataset
        """
        print(f'F1PredictorPipeline ==> engineer_features...')
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
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        print("F1PredictorPipeline is all done.")


if __name__ == "__main__":
    F1PredictorPipeline()
