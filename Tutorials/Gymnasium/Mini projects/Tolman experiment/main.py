from Train import Experiment
if __name__ == "__main__":

    group_1 = Experiment(group=1)
    group_1.Run()
    group_1.Plot_save(rolling_length=100)
    group_2 = Experiment(group=2)
    group_2.Run()
    group_2.Plot_save(rolling_length=100)
    group_3 = Experiment(group=3)
    group_3.Run()
    group_3.Plot_save(rolling_length=100)