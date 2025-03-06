import grid2op
import numpy as np

env_name = "rte_case14_realistic"
env = grid2op.make(env_name)
# /Users/chunyu/data_grid2op/rte_case14_realistic_val

val_chron, test_chron = np.load("./grid2op_env/train_val_test_split/val_chronics.npy"), \
 np.load("./grid2op_env/train_val_test_split/test_chronics.npy")

val_chron = np.array([str(i) for i in val_chron])
test_chron = np.array([str(i) for i in test_chron])

nm_env_train, m_env_val, nm_env_test = env.train_val_split(
                                  test_scen_id=test_chron, 
                                  add_for_test="test",
                                  val_scen_id=val_chron,
 )

env_train = grid2op.make(env_name+"_train")
env_val = grid2op.make(env_name+"_val")
env_test = grid2op.make(env_name+"_test")
