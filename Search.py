import numpy as np
from sklearn.metrics import balanced_accuracy_score
from deap import base, creator, tools, algorithms
import json
import tensorflow as tf
from tf.keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from Baseline import BaselineModel
import Utils

# Example search space
search_space = {
    'train_backbone': [True, False],
    'regularization': [None, 'l2'],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [4, 8],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
    'clip_length': [3, 5]
}

def random_indices():
    return [np.random.randint(len(search_space[key])) for key in search_space.keys()]

def evaluate_hyperparameters(individual, search_space):
    params = {key: search_space[key][value] for key, value in zip(search_space.keys(), individual)}

    # Build and train the Movinet Classifier model with the given hyperparameters
    model = BaselineModel(
                    model_id='a2', model_type="base", 
                    epochs=10, shots=15, 
                    dropout=params['dropout'], 
                    resolution=Utils.MOVINET_PARAMS['a2'][0], 
                    num_frames=int(2*np.floor(Utils.MOVINET_PARAMS['a2'][1]*params['clip_length']/2)), 
                    num_classes=len(Utils.LABEL_NAMES),
                    batch_size=params['batch_size'], 
                    frame_step=int(Utils.FPS/Utils.MOVINET_PARAMS['a2'][1]),
                    train_backbone=params['train_backbone'],
                    regularization=params['regularization'],
                    output_signature=Utils.OUTPUT_SIGNATURE,
                    label_names=Utils.LABEL_NAMES)
    
    model.init_data('.mp4', Utils.TRAIN_FOLDER, Utils.VAL_FOLDER, Utils.TEST_FOLDER)
    model.init_base_model()
    
    model.train(params['learning_rate'])

    # Evaluate the model on the validation data
    test = Utils.remove_paths(model.val_ds)
    actual, predicted = Utils.get_actual_predicted_labels(test, model.base_model)
    accuracy = balanced_accuracy_score(actual, predicted)

    K.clear_session()

    # Minimize the negative accuracy for the genetic algorithm
    return -accuracy,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Define how to create individuals and population
toolbox.register("indices", random_indices)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register evaluation, mating, mutation, and selection functions
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, high=len(search_space), indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("evaluate", evaluate_hyperparameters, search_space=search_space)

pop = toolbox.population(n=10)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats)

best_individual = tools.selBest(pop, 1)[0]
best_hyperparameters = {key: search_space[key][value] for key, value in zip(search_space.keys(), best_individual)}

results = {
    'best_hyperparameters': best_hyperparameters,
    'logbook': logbook.__dict__,
}

with open('genetic_algorithm_baseline.json', 'w') as outfile:
    json.dump(results, outfile)

