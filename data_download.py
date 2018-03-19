import argparse
import concurrent.futures
import logging
import os
import re
import requests
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm

from data_info import get_json_data


def download_image(url: str, folder: str, image_id: int, regexp):
    try:
        req = requests.get(url, timeout=60)
        if req.status_code == 200:
            file_extension = 'jpg'
            if 'Content-Type' in req.headers:
                searched = regexp.search(req.headers['Content-Type'])
                if searched is not None:
                    file_extension = searched.groups()[0]
            with open("{}/{:0>6}.{}".format(folder, image_id, file_extension), 'wb') as img_file:
                img_file.write(req.content)
            return
        else:
            raise Exception("Bad status: {}".format(req.status_code))
    except Exception as exc:
        error_message = "Image ID: {}; Error: {}".format(image_id, exc)
        logging.error(error_message)
        return


def download_set(json_path: str, save_folder: str, log_path: str):
    logging.basicConfig(filename=log_path, level=logging.ERROR)

    images, _ = get_json_data(json_path)
    reg = re.compile('image/(\w+)')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 8) as executor:
        futures = []
        for data in images:
            url_str = data['url'][0]
            img_id = data['image_id']
            futures.append(executor.submit(partial(download_image, url_str, save_folder, img_id, reg)))
        [future.result() for future in tqdm(futures)]
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    if args.train:
        download_set('data/train.json', './data/images/train', './logs/download-train.log')
    elif args.test:
        download_set('data/test.json', './data/images/test', './logs/download-test.log')
    elif args.validation:
        download_set('data/validation.json', './data/images/validation', './logs/download-validation.log')
    else:
        print('ooops!')
    return


if __name__ == "__main__":
    main()
