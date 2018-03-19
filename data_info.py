import collections
import json
import matplotlib.pyplot as plt

from params import NUM_CLASS


def get_json_data(path: str):
    with open(path) as json_data:
        data = json.load(json_data)
        imgs, annos = None, None
        if 'images' in data:
            imgs = data['images']
        if 'annotations' in data:
            annos = data['annotations']
        return imgs, annos


def get_class_distribution(annos: dict):
    if annos[0] and 'label_id' not in annos[0]:
        raise KeyError
    counter = collections.Counter()
    for ann_item in annos:
        counter[ann_item['label_id']] += 1
    return counter


def show_class_distribution_plot(data: dict):
    x_vals = list(data.keys())
    y_vals = list(data.values())
    x_labels = list(range(1, NUM_CLASS + 1, 9))
    plt.bar(x_vals, y_vals, align='center')
    plt.xticks(x_labels, x_labels)
    plt.show()


def main():
    _, annotations = get_json_data('data/train.json')

    if annotations is not None:
        class_dist = get_class_distribution(annotations)
        class_dist_sorted = dict(sorted(class_dist.items()))
        show_class_distribution_plot(class_dist_sorted)
        print(class_dist.most_common(10))
    return


if __name__ == "__main__":
    main()
