from metaflow import FlowSpec, step


class DataCollectionFlow(FlowSpec):
    """
    DataCollectionFlow is a flow that runs the data collection process.
    """

    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow.
        """
        print("DataCollectionFlow...")
        self.next(self.data_collection_step_one)

    @step
    def data_collection_step_one(self):
        """
        data_collection_step_one step
        """
        guest = 'Jose'
        print(f'DataCollectionFlow says: Hi {guest}!')
        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        print("DataCollectionFlow is all done.")


if __name__ == "__main__":
    DataCollectionFlow()
