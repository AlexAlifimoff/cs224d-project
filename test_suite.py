import network

def save_load_test():
    n = network.SummarizationNetwork()
    n.initialize()
    n.save("test_suite_1.net")
    m = network.SummarizationNetwork()
    m = m.load("test_suite_1.net")

    for np, mp in zip(n.params, m.params):
        assert((np.get_value() == mp.get_value()).all() )
    return True
    

def run_tests():
    save_load_test()


run_tests()
