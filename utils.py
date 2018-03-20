import collections
import csv
import json
import matplotlib.pyplot as plt
import os
import re

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


def get_missed_ids(log_file: str) -> set:
    missed = set()
    with open(log_file, 'r') as log:
        for line in log.readlines():
            findall = re.findall('Image ID: (\d{1,6})', line)
            findall_len = len(findall)
            if findall_len == 1:
                missed.add(int(findall[0]))
            elif findall_len > 0:
                missed.update(map(lambda x: int(x), findall))
    return missed


def get_existing_ids(log_file: str, original_set_size: int) -> set:
    """
        example: get_existing_ids('./logs/download-train.txt', TRAIN_SET_ORIGINAL_SIZE)
    """
    missed_ids = get_missed_ids(log_file)
    print('missed: ', len(missed_ids))
    original_ids = set(range(1, original_set_size + 1))
    return original_ids.difference(missed_ids)


def write_metadata(json_file: str, save_path: str, data_folder: str):
    """
        example: write_metadata('./data/validation.json', './data/validation_actual.csv', './data/images/validation/')
    """
    _, annotations = get_json_data(json_file)
    if annotations is None:
        raise Exception("These aren't the annotations you're looking for!")
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['filename', 'id', 'class'])
        print('files: ', len(os.listdir(data_folder)))
        for img_file in os.listdir(data_folder):
            img_id = int(re.search('^(\d{6})\.', img_file).groups()[0])
            if annotations[img_id - 1]:
                writer.writerow([img_file, img_id, annotations[img_id - 1]['label_id']])
    return


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
