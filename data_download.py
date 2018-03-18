import concurrent.futures
import re
import requests
import traceback
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
            return None
        else:
            raise Exception("Bad status: {}".format(req.status_code))
    except Exception as exc:
        error_message = "Image ID: {}; Error: {}; Trace: {}".format(image_id, exc, traceback.format_exc())
        return error_message


images, _ = get_json_data('data/train.json')
reg = re.compile('image/(\w+)')

with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
    save_path = './data/images/train'
    futures = []
    for data in images:
        url_str = data['url'][0]
        img_id = data['image_id']
        futures.append(executor.submit(partial(download_image, url_str, save_path, img_id, reg)))
    results = [future.result() for future in tqdm(futures)]

errors = [r for r in results if r is not None]
if len(errors) > 0:
    with open('./logs/download-train.txt', 'w') as log_file:
        log_file.writelines(errors)
