import argparse
from collections import defaultdict
import json

def evaluate(instances):
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for instance in instances:
        confusion_matrix[instance["predicted_label"]][instance["gold_label"]] += 1

    classes = sorted(confusion_matrix.keys())
    max_class = max(6, len(max(classes, key=len)))

    precision = dict()
    recall = dict()
    f1 = dict()

    confusion_matrix_string = ['\n', ' ' * max_class]
    for c in classes:
        confusion_matrix_string.append('\t')
        confusion_matrix_string.append(c)
        confusion_matrix_string.append(' ' * (max_class - len(c)))
    confusion_matrix_string.append('\n')
    for c1 in classes:
        confusion_matrix_string.append(c1)
        confusion_matrix_string.append(' ' * (max_class - len(c1)))
        for c2 in classes:
            confusion_matrix_string.append('\t')
            ct = str(confusion_matrix[c1][c2])
            confusion_matrix_string.append(ct)
            confusion_matrix_string.append(' ' * (max_class - len(ct)))
        confusion_matrix_string.append('\n')
        precision[c1] = confusion_matrix[c1][c1] / max(1.0, sum(p[c1] for p in confusion_matrix.values()))
        recall[c1] = confusion_matrix[c1][c1] / max(1.0, sum(confusion_matrix[c1].values()))
        f1[c1] = 2 * precision[c1] * recall[c1] / max(1.0, precision[c1] + recall[c1])

    accuracy = sum(confusion_matrix[c][c] for c in classes) / max(
        1.0, sum(sum(vs.values()) for vs in confusion_matrix.values()))

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': ''.join(confusion_matrix_string)
    }

def pretty_print_results(d, prefix=''):
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        if isinstance(v, dict):
            print(prefix + k + ":")
            pretty_print_results(v, prefix + '\t')
        elif '\n' in str(v):
            print(prefix + k + ":")
            print(str(v).replace('\n', '\n' + prefix + '\t'))
        else:
            print(prefix + k + ":", str(v))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str)
    parser.add_argument("--score_file",type=str)
    args = parser.parse_args()

    with open(args.prediction_file, 'r') as f:
        instances = [json.loads(line) for line in f]
    
    res = evaluate(instances)
    pretty_print_results(res)
    
    print('score file is saved at {}'.format(args.score_file))
    with open(args.score_file, "w") as f:
        f.write(json.dumps(res))