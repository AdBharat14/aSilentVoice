from firebaseAPI import post_data, post_summary, get_summary
import easyocr
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import google.generativeai as palm
from PIL import Image
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
import os
from dataclasses import dataclass
from typing import Callable
from tqdm import tqdm
import numpy as np

palm.configure(api_key='AIzaSyCY3YN9Xucx91p1IdIuVtCkGaKbZCm0gD4')
dialog_detector = YOLO("Dialog_detector.pt","v8")
person_classifier = YOLO("PersonClassifier.pt","v8")
person_detector = YOLO("person_detector.pt","v8")


dic = {}
lis = []
VAR = 0

supported_types = [
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".pbm",
    ".pgm",
    ".pp",
    ".pxm",
    ".pnm",
    ".pfm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".exr",
    ".hdr",
    ".pic",
]


@dataclass
class ImageWithFilename:
    image: np.ndarray
    image_name: str


def get_file_names(directory_path: str) -> list[str]:
    """
    Returns the names of the files in the given directory
    """
    if not os.path.exists(directory_path):
        return []
    return [
        file_name
        for file_name in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, file_name))
    ]


def load_image(directory_path: str, image_name: str) -> ImageWithFilename:
    """
    Returns a ImageWithFilename object from the given image name in the given directory
    """
    image = cv2.imread(os.path.join(directory_path, image_name))
    return ImageWithFilename(image, image_name)


def get_file_extension(file_path: str) -> str:
    """
    Returns the extension of the given file path
    """
    return os.path.splitext(file_path)[1]


def load_images(directory_path: str) -> list[ImageWithFilename]:
    """
    Returns a list of ImageWithFilename objects from the images in the given directory
    """
    file_names = get_file_names(directory_path)
    image_names = filter(lambda x: get_file_extension(x) in supported_types, file_names)
    return [load_image(directory_path, image_name) for image_name in image_names]


def get_background_intensity_range(grayscale_image: np.ndarray, min_range: int = 1) -> tuple[int, int]:
    """
    Returns the minimum and maximum intensity values of the background of the image
    """
    edges = [grayscale_image[-1, :], grayscale_image[0, :], grayscale_image[:, 0], grayscale_image[:, -1]]
    sorted_edges = sorted(edges, key=lambda x: np.var(x))
    # print(sorted_edges[0])
    max_intensity = max(sorted_edges[0])
    min_intensity = max(min(min(sorted_edges[0]), max_intensity - min_range), 0)

    return min_intensity, max_intensity



def is_contour_rectangular(contour: np.ndarray) -> bool:
    """
    Returns whether the given contour is rectangular or not
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(approx) == 4:
        global dic,lis,VAR
        # tempList = sorted(contour, key = lambda x: x[0][0]+x[0][1])
        # x = list(map(str,tempList[-1][0]))

        # dic[','.join(list(map(str,tempList[-1][0])))] = VAR
        # VAR+=1
        # lis.append(list(map(int,tempList[-1][0])))
        # print(tempList[-1])
        # print("helllooooooooo")
        return True
    return False


def generate_background_mask(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Generates a mask by focusing on the largest area of white pixels
    """
    WHITE = 255
    LESS_WHITE, _ = get_background_intensity_range(grayscale_image, 25)
    LESS_WHITE = max(LESS_WHITE, 240)

    ret, thresh = cv2.threshold(grayscale_image, LESS_WHITE, WHITE, cv2.THRESH_BINARY)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh)

    mask = np.zeros_like(thresh)

    PAGE_TO_SEGMENT_RATIO = 1024

    halting_area_size = mask.size // PAGE_TO_SEGMENT_RATIO

    mask_height, mask_width = mask.shape
    base_background_size_error_threshold = 0.05
    whole_background_min_width = mask_width * (1 - base_background_size_error_threshold)
    whole_background_min_height = mask_height * (1 - base_background_size_error_threshold)

    for i in np.argsort(stats[1:, 4])[::-1]:
        x, y, w, h, area = stats[i + 1]
        if area < halting_area_size:
            break
        if (
            (w > whole_background_min_width) or
            (h > whole_background_min_height) or
            (is_contour_rectangular(cv2.findContours((labels == i + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]))
        ):
            mask[labels == i + 1] = WHITE

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    return mask


def extract_panels(
    image: np.ndarray,
    panel_contours: list[np.ndarray],
    accept_page_as_panel: bool = True,
) -> list[np.ndarray]:
    """
    Extracts panels from the image using the given contours corresponding to the panels
    """
    PAGE_TO_PANEL_RATIO = 32

    height, width = image.shape[:2]
    image_area = width * height
    area_threshold = image_area // PAGE_TO_PANEL_RATIO

    returned_panels = []

    for contour in panel_contours:
        global lis,dic,VAR
        x, y, w, h = cv2.boundingRect(contour)

        if not accept_page_as_panel and ((w >= width * 0.99) or (h >= height * 0.99)):
            continue

        area = cv2.contourArea(contour)

        if (area < area_threshold):
            continue
        dic[','.join(list(map(str,[x,y,y+h,x+w])))] = VAR
        VAR+=1
        # lis.append(list(map(int,[y+h,x+w])))
        lis.append([x,y,x+w,y+h])
        fitted_panel = image[y: y + h, x: x + w]

        returned_panels.append(fitted_panel)
    # with open("myfile.txt", "w") as file1:
    #     for x in returned_panels:
    #         file1.write('[')
    #         for y in x:
    #             file1.write('[')
    #             for z in y:
    #                 file1.write('[')
    #                 for k in z:
    #                     file1.write(str(k))
    #                     file1.write(',')
    #                 file1.write(']')
    #             file1.write(']')
    #         file1.write(']')
    return returned_panels


def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Applies adaptive threshold to the given image
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)


def generate_panel_blocks(
        image: np.ndarray, 
        background_generator: Callable[[np.ndarray], np.ndarray] = generate_background_mask,
        split_joint_panels: bool = False,
        fallback: bool = True,
) -> list[np.ndarray]:
    """
    Generates the separate panel images from the base image
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    processed_image = cv2.Laplacian(processed_image, -1)
    processed_image = cv2.dilate(processed_image, np.ones((5, 5), np.uint8), iterations=1)
    processed_image = 255 - processed_image

    mask = background_generator(processed_image)

    STRIPE_FORMAT_MASK_AREA_RATIO = 0.3
    mask_area = np.count_nonzero(mask)
    mask_area_ratio = mask_area / mask.size

    if STRIPE_FORMAT_MASK_AREA_RATIO > mask_area_ratio and split_joint_panels:
        pixels_before = np.count_nonzero(mask)
        mask = cv2.ximgproc.thinning(mask)
        
        up_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
        down_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
        left_kernel = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], np.uint8)
        right_kernel = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], np.uint8)

        down_right_kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
        down_left_diagonal_kernel = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], np.uint8)
        up_left_diagonal_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], np.uint8)
        up_right_diagonal_kernel = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], np.uint8)
        
        PAGE_TO_JOINT_OBJECT_RATIO = 3
        image_height, image_width = grayscale_image.shape

        height_based_size = image_height // PAGE_TO_JOINT_OBJECT_RATIO
        width_based_size = (2 * image_width) // PAGE_TO_JOINT_OBJECT_RATIO

        height_based_size += height_based_size % 2 + 1
        width_based_size += width_based_size % 2 + 1

        up_dilation_kernel = np.zeros((height_based_size, height_based_size), np.uint8)
        up_dilation_kernel[height_based_size // 2:, height_based_size // 2] = 1

        down_dilation_kernel = np.zeros((height_based_size, height_based_size), np.uint8)
        down_dilation_kernel[:height_based_size // 2 + 1, height_based_size // 2] = 1

        left_dilation_kernel = np.zeros((width_based_size, width_based_size), np.uint8)
        left_dilation_kernel[width_based_size // 2, width_based_size // 2:] = 1

        right_dilation_kernel = np.zeros((width_based_size, width_based_size), np.uint8)
        right_dilation_kernel[width_based_size // 2, :width_based_size // 2 + 1] = 1

        min_based_size = min(width_based_size, height_based_size)

        down_right_dilation_kernel = np.identity(min_based_size // 2 + 1, dtype=np.uint8)
        down_right_dilation_kernel = np.pad(down_right_dilation_kernel, ((0, min_based_size // 2), (0, min_based_size // 2)))

        up_left_dilation_kernel = np.identity(min_based_size // 2 + 1, dtype=np.uint8)
        up_left_dilation_kernel = np.pad(up_left_dilation_kernel, ((min_based_size // 2, 0), (0, min_based_size // 2)))

        up_right_dilation_kernel = np.flip(np.identity(min_based_size // 2 + 1, dtype=np.uint8), axis=1)
        up_right_dilation_kernel = np.pad(up_right_dilation_kernel, ((min_based_size // 2, 0), (0, min_based_size // 2)))

        down_left_dilation_kernel = np.flip(np.identity(min_based_size // 2 + 1, dtype=np.uint8), axis=1)
        down_left_dilation_kernel = np.pad(down_left_dilation_kernel, ((0, min_based_size // 2), (min_based_size // 2, 0)))

        match_kernels = [
            up_kernel,
            down_kernel,
            left_kernel,
            right_kernel,
            down_right_kernel,
            down_left_diagonal_kernel,
            up_left_diagonal_kernel,
            up_right_diagonal_kernel,
        ]

        dilation_kernels = [
            up_dilation_kernel,
            down_dilation_kernel,
            left_dilation_kernel,
            right_dilation_kernel,
            down_right_dilation_kernel,
            down_left_dilation_kernel,
            up_left_dilation_kernel,
            up_right_dilation_kernel,
        ]

        def get_dots(grayscale_image: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, int]:
            temp = cv2.matchTemplate(grayscale_image, kernel, cv2.TM_CCOEFF_NORMED)
            _, temp = cv2.threshold(temp, 0.9, 1, cv2.THRESH_BINARY)
            temp = np.where(temp == 1, 255, 0).astype(np.uint8)
            pad_height = (kernel.shape[0] - 1) // 2
            pad_width = (kernel.shape[1] - 1) // 2
            temp = cv2.copyMakeBorder(temp, pad_height, kernel.shape[0] - pad_height - 1, pad_width, kernel.shape[1] - pad_width - 1, cv2.BORDER_CONSTANT, value=0)
            return temp
        
        for match_kernel, dilation_kernel in zip(match_kernels, dilation_kernels):
            dots = get_dots(mask, match_kernel)
            lines = cv2.dilate(dots, dilation_kernel, iterations=1)
            mask = cv2.bitwise_or(mask, lines)

        pixels_now = np.count_nonzero(mask)
        dilation_size = pixels_before // (4  * pixels_now)
        dilation_size += dilation_size % 2 + 1
        mask = cv2.dilate(mask, np.ones((dilation_size, dilation_size), np.uint8), iterations=1)

        page_without_background = 255 - mask
    else:
        page_without_background = cv2.subtract(grayscale_image, mask)
            
    contours, _ = cv2.findContours(page_without_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # with open('myfile.txt', 'w') as f:
    #     simplejson.dump(contours, f)
    # print(type(contours))
    # print(type(contours[1]))
    # print(contours[0])
    # temp=0
    # for x in contours:
    #     print(x)
    #     print()
    #     print()
    #     print()
    #     print()
    #     temp+=1
    #     if(temp>10):
    #         break
    # print()
    # print()
    panels = extract_panels(image, contours)
    # with open("myfile.txt", "a") as file1:
    #     for x in contours:
    #         file1.write('[')
    #         for y in x:
    #             file1.write('[')
    #             for z in y:
    #                 file1.write('[')
    #                 for k in z:
    #                     file1.write(str(k))
    #                     file1.write(',')
    #                 file1.write(']')
    #             file1.write(']')
    #         file1.write(']')
    if fallback and len(panels) < 2:
        processed_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
        processed_image = cv2.Laplacian(processed_image, -1)
        _, thresh = cv2.threshold(processed_image, 8, 255, cv2.THRESH_BINARY)
        processed_image = apply_adaptive_threshold(processed_image)
        processed_image = cv2.subtract(processed_image, thresh)
        processed_image = cv2.dilate(processed_image, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort(contours)
        # with open("myfile.txt", "w") as file1:
        #     for x in contours:
        #         file1.write('[')
        #         for y in x:
        #             file1.write('[')
        #             for z in y:
        #                 file1.write('[')
        #                 for k in z:
        #                     file1.write(str(k))
        #                     file1.write(',')
        #                 file1.write(']')
        #             file1.write(']')
        #         file1.write(']')
        # print(contours)
        panels = extract_panels(image, contours, False)
    # with open("myfile.txt", "w") as file1:
    #     for x in panels:
    #         file1.write('[')
    #         for y in x:
    #             file1.write('[')
    #             for z in y:
    #                 file1.write('[')
    #                 for k in z:
    #                     file1.write(str(k))
    #                     file1.write(',')
    #                 file1.write(']')
    #             file1.write(']')
    #         file1.write(']')
    # print(panels)
    return panels


def extract_panels_for_image(image_path: str, output_dir: str, fallback: bool = True, split_joint_panels: bool = False):
    """
    Extracts panels for a single image
    """
    if not os.path.exists(image_path):
        return
    image_path = os.path.abspath(image_path)
    image = load_image(os.path.dirname(image_path), image_path)
    image_name, image_ext = os.path.splitext(image.image_name)
    panel_blocks = generate_panel_blocks(image.image, split_joint_panels=split_joint_panels, fallback=fallback)
    for k, panel in enumerate(tqdm(panel_blocks, total=len(panel_blocks))):
        out_path = os.path.join(output_dir, f"{image_name}_{k}{image_ext}")
        cv2.imwrite(out_path, panel)


def extract_panels_for_images_in_folder(input_dir: str, output_dir: str, fallback: bool = True, split_joint_panels: bool = False):
    """
    Basically the main function of the program,
    this is written with cli usage in mind
    """
    if not os.path.exists(output_dir):
        return
    files = os.listdir(input_dir)
    num_files = len(files)
    for i, image in enumerate(tqdm(load_images(input_dir), total=num_files)):
        global dic,VAR,lis
        # print("hello")
        image_name, image_ext = os.path.splitext(image.image_name)
        dic = {}
        lis = []
        VAR = 0
        images = generate_panel_blocks(image.image, fallback=fallback, split_joint_panels=split_joint_panels)
        # print(lis)
        global_lis = []
        # def sort_verDiv(x):
            
        # def sort_hor_div(i,x):
        #     lst = []
        #     lst.append(lst[0])
        #     x,y = lst[0]
        #     x,y = lst[0]

        
        def customSort(a):
            n = len(a)

            adj = [set() for _ in range(n)]

            for i in range(n):
                sx, sy, ex, ey = a[i]
                for j in range(n):
                    if j == i:
                        continue
                    sjx, sjy, ejx, ejy = a[j]
                    if sjy > ey:
                        adj[i].add(j)
                    if sx > ejx and sy < ejy:
                        adj[i].add(j)

            indegree = [0] * n
            for i in range(n):
                for j in adj[i]:
                    indegree[j] += 1

            q = deque()
            for i in range(n):
                if indegree[i] == 0:
                    q.append(i)

            result = []
            while q:
                node = q.popleft()
                result.append(node)

                for it in adj[node]:
                    indegree[it] -= 1
                    if indegree[it] == 0:
                        q.append(it)

            return result
        # print(lis)
        lis = customSort(lis)
        lis2 = lis.copy()
        index=0
        for temp in lis:
            lis2[temp]=index
            index+=1
        # for x in lis:
        #     for y in x:
        #         print(y,end=" ")
        # print(lis2)
        # dic2={}
        # ind = 0
        # for x in lis:
        #     y = dic.get(','.join(list(map(str,x))))
        #     dic2[y]=ind
        #     ind+=1
        # print(dic2)
        for j, panel in enumerate(images):
            # print("hi")
            # print(j)
            # print(panel)
            # print()
            # print()
            # print()
            # print()
            # print()
            # print()
            # print()
            # out_path = os.path.join(output_dir, f"{image_name}_{dic2.get(j)}{image_ext}")
            out_path = os.path.join(output_dir, f"{image_name}_{lis2[j]}{image_ext}")
            cv2.imwrite(out_path, panel)


class ExtractionThread(QThread):
    progress_update = pyqtSignal(str)
    process_finished = pyqtSignal()

    def __init__(self, input_dir: str, output_dir: str, split_joint_panels: bool = False, fallback: bool = True):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split_joint_panels = split_joint_panels
        self.fallback = fallback

    def run(self):
        files = os.listdir(self.input_dir)
        total_files = len(files)
        for i, image in enumerate(load_images(self.input_dir)):
            if self.isInterruptionRequested():
                return
            self.progress_update.emit(f"Processing file {i+1}/{total_files}")
            image_name, image_ext = os.path.splitext(image.image_name)
            for k, panel in enumerate(generate_panel_blocks(image.image, split_joint_panels=self.split_joint_panels, fallback=self.fallback)):
                out_path = os.path.join(self.output_dir, f"{image_name}_{k}{image_ext}")
                cv2.imwrite(out_path, panel)
        self.process_finished.emit()


def createPanels():
    
    # if 'output':
    extract_panels_for_images_in_folder('input', 'middle', None, None)
    # else:
    #     extract_panels_for_image('input', os.path.dirname('input'), None, None)





def create_story(manga, total_people, characters, dialogues):
    response = get_summary(manga)
    
    prompt = f'Now we want you to create a very small story.'
    prompt = f'There are {total_people} people in this setting. There are {len(characters)} main characters in this panel:\n'
    for x in characters:
        prompt+=x+"\n"
    prompt += f'There are {len(dialogues)} dialogues:\n'
    for x in dialogues:
        prompt += x + "\n"
    if response and len(response)!=0:
        prompt+= f'As the previous part of the story it goes as follows:\n {response}'
    # print(prompt)
    response = palm.chat(messages=prompt)
    post_data(manga, response.last)
    prompt+="Summarize the new part of the story in 5-10 words"
    response = palm.chat(messages = prompt)
    post_summary(manga, response.last)


def crop_rectangle(image_path, x1, y1, x2, y2, output_path):
    """
    Crop a rectangle from the given image and save it to the output path.

    Args:
        image_path (str): Path to the input image file.
        x1, y1, x2, y2 (int): Coordinates of the top-left and bottom-right corners of the rectangle to crop.
        output_path (str): Path to save the cropped image.
    """
    # Open the image
    image = Image.open(image_path)

    # Crop the rectangle
    cropped_image = image.crop((x1, y1, x2, y2))

    # Save the cropped image
    cropped_image.save(output_path)


def ocr(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    res = []
    for temp in result:
        res.append(temp[1])
    return ' '.join(res)


def classify(img_src):
    detection_output = person_classifier.predict(source=img_src , conf=0.25,save=False)
    conf = detection_output[0].numpy().probs.top1conf
    person = "Person"

    if conf > 0.8:
        person = detection_output[0].names[detection_output[0].numpy().probs.top1]
    
    return person


def findPeople(img_src):
    detection_output = person_detector.predict(source=img_src , conf=0.25,save=False)
    cnt = 0
    person = set()
    for i in detection_output[0].boxes.xyxy.numpy():
        x1 , y1 , x2 , y2 = i
        path = f"person.png"
        crop_rectangle(img_src, x1, y1, x2, y2, path)
        person.add(classify(path))
        cnt+=1
    return (cnt,person)


def findDialogue(img_src):
    segments = dialog_detector.predict(img_src , conf=0.25,save=False)
    cnt = 0
    dialogues = []
    for i in segments[0].boxes.xyxy.numpy():
        x1 , y1 , x2 , y2 = i
        path = f"dialogue.png"
        print(path)
        crop_rectangle(img_src, x1, y1, x2, y2, path)
        dialogues.append(ocr(path))
        cnt += 1
    print(cnt)
    # conf = segments[0].numpy().probs.top1conf
    # person = "Person"

    # if conf > 0.8:
    #     person = segments[0].names[detection_output[0].numpy().probs.top1]

    # print(person)
    #####find segments and store them in `segments`#####
    ################
    # for i in segments:
    return dialogues



def processImage(input_dir):
    files = os.listdir(input_dir)
    num_files = len(files)
    directory = './middle'
    file_names = os.listdir(directory)
    page = []
    for file_name in file_names:
        path = directory+'/'+file_name
        people = findPeople(path)
        print(people)
        # people[1].remove("Person")
        dialogues = findDialogue(path)
        print(dialogues)
        page.append([people[0],people[1],dialogues])
    create_story('asilentvoice', people[0], people[1], dialogues)

if __name__ == "__main__":
    createPanels()
    processImage('middle')