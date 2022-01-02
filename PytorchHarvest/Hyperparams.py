#we suppose num_steps = 300 and max_episodes = 100
def get_first_batch():
    # hyperparameters
    hidden_size = 32  # 256
    learning_rate = 0.01  # 1e-2

    return hidden_size, learning_rate
def get_second_batch():
    # hyperparameters
    hidden_size = 128  # 256
    learning_rate = 0.01  # 1e-2

    return hidden_size, learning_rate
#this seams to have interristing values
def get_third_batch():
    # hyperparameters
    hidden_size = 18  # 256
    learning_rate = 0.00136

    return hidden_size, learning_rate
def get_papers_batch():
    # hyperparameters
    hidden_size = 32  # 256
    learning_rate = 0.00136

    return hidden_size, learning_rate