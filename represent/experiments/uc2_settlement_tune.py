
from bayes_opt import BayesianOptimization
from represent.experiments.uc2_settlement_evaluation import prepare_data, run_finetune, run_maml, run_moco, run_randominit, run_supervised, evaluate
import numpy as np

def get_test_parameters(name):
    X, Y, x_test, existing_labels, buildings, meta = prepare_data()

    if name in ["supervised", "moco", "randominit"]:

        run_functions = {
            "supervised":run_supervised,
            "moco": run_moco,
            "randominit": run_randominit
        }

        def test_parameters(learning_rate_exp, epochs, momentum, weight_decay_exp):
            #X, Y, x_test, existing_labels, buildings, meta = prepare_data()

            args = dict(
                learning_rate=10**learning_rate_exp,
                epochs=int(epochs),
                momentum=momentum,
                weight_decay=10**weight_decay_exp
            )

            probability = run_functions[name](X, Y, x_test, args=args)

            ap, iou, optimal_threshold = evaluate(probability, buildings, mask=existing_labels, threshold=None)

            return ap

        return test_parameters

    elif name == "maml":

        def test_parameters(gradient_steps, inner_step_size):

            args = dict(
                inner_step_size=inner_step_size,
                gradient_steps=int(gradient_steps),
            )

            probability = run_maml(X, Y, x_test, args=args)

            ap, iou, optimal_threshold = evaluate(probability, buildings, mask=existing_labels, threshold=None)

            return ap

        return test_parameters

def get_pbounds(name):
    if name in ["supervised", "moco", "randominit"]:
        # Bounded region of parameter space
        pbounds = dict(
            learning_rate_exp=(-4, -1),
            epochs=(5,50),
            momentum=(0.9, 0.99),
            weight_decay_exp=(-10,-1)
        )
    elif name=="maml":
        pbounds = dict(
            inner_step_size=(0.1, 0.75),
            gradient_steps=(200, 3000),
        )

    return pbounds

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--n_iter',  type=int, default=5)
    parser.add_argument('--init_points', type=int, default=2)

    args = parser.parse_args()

    name = args.name
    n_iter = args.n_iter
    init_points = args.init_points

    test_parameters = get_test_parameters(name)
    pbounds = get_pbounds(name)

    optimizer = BayesianOptimization(
        f=test_parameters,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    import json
    import os
    import pandas as pd

    maxfilename = os.path.join("/data/RepreSent/UC2/uc2_settlements/tune", f"tune_{name}.json")
    csvfilename = os.path.join("/data/RepreSent/UC2/uc2_settlements/tune", f"tune_{name}.csv")

    stats = []
    for result in optimizer.res:
        target = result["target"]
        stat = result["params"]
        stat["target"] = target
        stats.append(stat)
    pd.DataFrame(stats).to_csv(csvfilename)

    json_string = json.dumps(optimizer.max)
    with open(maxfilename, "w") as f:
        f.write(json_string)

