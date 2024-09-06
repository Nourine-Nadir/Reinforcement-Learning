# from Train_Q_Table import Experiment
from Train_DQN import Experiment
def run_experiments(groups):
    for group in groups:
        let = Experiment(group=group)
        let.Run()
        let.Plot_save(rolling_length=50)
    return
if __name__ == "__main__":

    run_experiments([1,2,3])
