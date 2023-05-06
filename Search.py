import numpy as np
from sklearn.metrics import balanced_accuracy_score
from deap import base, creator, tools, algorithms
import json
from tensorflow.keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Baseline import BaselineModel
from FewShotLearning import FewShotModel
import Utils


class GeneticSearchBaseline():
    
    def __init__(self, search_space, n_gen=10, n_pop=6):
        self.search_space = search_space
        self.n_gen = n_gen
        self.n_pop = n_pop

    def random_indices(self):
        return [np.random.randint(len(self.search_space[key])) for key in self.search_space.keys()]

    def get_fitness_values(self, ind):
        return ind.fitness.values
    
    def init_model(self, params):
        return BaselineModel(
                        model_id='a2', model_type="base", 
                        epochs=5, shots=10, 
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

    def evaluate_hyperparameters(self, individual):
        params = {key: self.search_space[key][value] for key, value in zip(self.search_space.keys(), individual)}

        # Build and train the Movinet Classifier model with the given hyperparameters
        model = self.init_model(params)
        
        model.init_data('.mp4', Utils.TRAIN_FOLDER, Utils.VAL_FOLDER, Utils.TEST_FOLDER)
        if model.name == 'FSL':
            model.init_meta_data('.avi', Utils.META_TRAIN_FOLDER, Utils.META_VAL_FOLDER)
        model.init_base_model()
        
        model.train(params['learning_rate'])

        # Evaluate the model on the validation data
        test = Utils.remove_paths(model.val_ds)
        actual, predicted = Utils.get_actual_predicted_labels(test, model.base_model)
        accuracy = balanced_accuracy_score(actual, predicted)

        K.clear_session()

        # Minimize the negative accuracy for the genetic algorithm
        return -accuracy,

    def run(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Define how to create individuals and population
        toolbox.register("indices", self.random_indices)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register evaluation, mating, mutation, and selection functions
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=[len(val) - 1 for val in self.search_space.values()], indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate_hyperparameters)

        pop = toolbox.population(n=self.n_pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(self.get_fitness_values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        for gen in range(self.n_gen):
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=stats, verbose=False)
            hof.update(pop)  # Add this line to update the Hall of Fame
            best_individual = hof[0]
            best_hyperparameters = {key: self.search_space[key][value] for key, value in zip(self.search_space.keys(), best_individual)}

            # Print the entire population and their fitness values
            print(f"Generation {gen+1}, Population:")
            for idx, individual in enumerate(pop):
                print(f"  Individual {idx+1}: {individual}, Fitness: {individual.fitness.values[0]}")

            # Print the best hyperparameters found so far
            print(f"Generation {gen+1}, Best hyperparameters: {best_hyperparameters}")

            # Print the logbook for this generation
            print(f"Generation {gen+1}, Logbook: {logbook}")
        
        results = {
            'best_hyperparameters': best_hyperparameters,
            'logbook': logbook.__dict__,
        }
        print(results)
        with open(f'genetic_algorithm_baseline_{N_POP}_{N_GEN}.json', 'w') as outfile:
            json.dump(results, outfile)

class GeneticSearchFSL(GeneticSearchBaseline):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def init_model(self, params):
        return FewShotModel(
                        tasks=params['meta_tasks'], meta_classes=Utils.META_CLASSES,
                        model_id='a2', model_type="base", 
                        epochs=5, shots=params['shots'], 
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

if __name__ == '__main__':
    search_space_baseline = {
        'train_backbone': [True, False],
        'regularization': [None, 'l2'],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [4, 8],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'clip_length': [3, 5]
    }
    search_space_fsl = {
        'train_backbone': [True, False],
        'regularization': [None, 'l2'],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [4, 8],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'clip_length': [3, 5],
        'meta_tasks': [5, 10, 15],
        'shots': [3, 5, 10]
    }
    
    gsb = GeneticSearchBaseline(search_space_baseline, 1, 1)
    gsfsl = GeneticSearchFSL(search_space_fsl, 1, 1)
    
    gsb.run()