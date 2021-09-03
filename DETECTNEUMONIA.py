import os
import cv2
import sys
import streamlit as st
import colorsys
from skimage.measure import find_contours
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage.draw
import random
import skimage.draw
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import skimage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import patches, lines
import cv2
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import streamlit.components.v1 as components
import base64
from io import BytesIO
import imutils
import copy
import time

ROOT_DIR = 'C:/Users/Asus/Documents/Tesis/MaskRCNN_Video-master'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

sys.path.append(ROOT_DIR)

from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_neumonia_0130.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

min_confidence = 0.95
# Clases
names = ['NoNeumonia', 'Neumonia']

class_names = ['BG', 'nodulo pulmonar']
real_test_dir = 'C:/Users/Asus/Documents/Tesis/MaskRCNN_Video-master/test/'
min_confidence = 0.95


class neumoniaConfig(Config):
    """Configuration for training on the helmet  dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nodulo pulmonar"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (casco)

    # All of our training images are 512x512
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 400

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 100

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000


config = neumoniaConfig()
config.display()


class InferenceConfig(neumoniaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = min_confidence


inference_config = InferenceConfig()
# source code from visulaize.py


##guardar imagen eejcutar display
def display2_instances_cust(image, boxes, masks, class_ids, class_names,
                           scores=None, title="",
                           figsize=(12, 12), ax=None,
                           show_mask=True, show_bbox=True,
                           colors=None, captions=None, imagecount=0):
    # Number of instances
    N = boxes.shape[0]
    inicio2 = time.time()
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 0, -0)
    ax.set_xlim(-0, width + 0)
    ax.axis('off')
    ax.set_title(title)

    #masked_image = image.astype(np.uint32).copy()
    #masked_image2 = image.astype(np.uint32).copy()
    masked_image=copy.copy(image)
    masked_image2=copy.copy(image)

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        color_r = [255, 0, 0]
        masked_image2[y1:y1 + 2, x1:x2] = color_r
        masked_image2[y2:y2 + 2, x1:x2] = color_r
        masked_image2[y1:y2, x1:x1 + 2] = color_r
        masked_image2[y1:y2, x2:x2 + 2] = color_r

        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{}".format(label)
        # ax.text(x1, y1 + 8, caption,                               #if not showing text
        #        color='b', size=20, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        # if show_mask:
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor='none')
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(f"C:/Users/Asus/Documents/Tesis/MaskRCNN_Video-master/hola.jpg", bbox_inches='tight',
                transparent=True, pad_inches=-0.5,
                orientation='landscape')  # save output

    if auto_show:
        # plt.show()
        def get_image_download_link(img, filename, text):
            """Generates a link allowing the PIL image to be downloaded
            in:  PIL image
            out: href string
            """
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
            return href

        if len(class_ids) != 0:
            Predi_imag= cv2.hconcat([masked_image, masked_image2])

            result = Image.fromarray(Predi_imag)
            st.markdown(get_image_download_link(result, 'Paciente con neumonia.JPEG','💾'+ ' Descargar ' + ' resultado'),
                        unsafe_allow_html=True)

            st.header('IMAGEN SEGMENTADA')
            # masked_image = imutils.resize(masked_image, width=600)
            st.image(masked_image, use_column_width=100)
            st.header('IMAGEN BOXES')
            # masked_image2 = imutils.resize(masked_image2, width=600)
            st.image(masked_image2, use_column_width=100)
            # imagen_conc = cv2.vconcat([masked_image,masked_image2])

        else:
            result = Image.fromarray(masked_image)
            st.markdown(get_image_download_link(result, 'Paciente sin neumonia.JPEG','💾'+' Descargar ' + ' resultado'),
                        unsafe_allow_html=True)
            st.header('IMAGEN')
            st.image(masked_image, use_column_width=300)
        fin2 = time.time()
        tiempo2=fin2-inicio2
    return tiempo2



def get_image_download_link(img, filename, text):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    return image



def main():
    st.set_page_config(page_title = "Predictor neumonia",page_icon = '✅', layout = "wide", initial_sidebar_state = "expanded")
    # Recreate the model in inference mode
    model_filename = "mask_rcnn_neumonia_0130.h5"
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='models')


    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join('models', model_filename)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    image = Image.open('logo.png')
    st.sidebar.image(image,caption=None, width=200, use_column_width='always', clamp=False, channels='RGB', output_format='auto')
    st.title("CLASIFICADOR DE PACIENTES CON NEUMONIA EN IMAGENES RX")
    predictS = ""
    # img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
    uploaded_image = st.sidebar.file_uploader('🗂️ '+"Cargar una imagen de RX frontal:", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        inicio3 = time.time()
        with st.spinner('📤 '+'Subiendo imagen...'):
            time.sleep(1)
        st.sidebar.success('📁 '+'Imagen subida')
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fin3 = time.time()
        tiempo3=fin3-inicio3



    # El usuario carga una imagen
    # if img_file_buffer is not None:
    # image = np.array(Image.open(img_file_buffer))
    # st.image(image, caption="Imagen", use_column_width=False)

    # El botón predicción se usa para iniciar el procesamiento

    if st.sidebar.button('✅ '+'Predicción'):
      inicio = time.time()
      if uploaded_image:
        results = model.detect([image], verbose=1)
        r = results[0]
        with st.spinner('📝 '+'Procesando imagen...'):
             time.sleep(1)

        if len(r['class_ids']) != 0:
            label = "PACIENTE CON NEUMONIA"
            color = (255, 0, 0)

        else:
            label = "PACIENTE SIN NEUMONIA"
            color = (0, 255, 0)
        st.success('EL DIAGNÓSTICO ES: '+label)
        cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 4)

        N = len(r['class_ids'])
        for i in range(N):
           color2 = [0, 255, 0]
           score = r['scores'][i]
           score = str(score)
           y1, x1, y2, x2 = r['rois'][i]
           cv2.putText(image, score, (x1, y1-2 ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color2, 2)

        fin = time.time()
        tiempo1=fin-inicio
        tiempo2=display2_instances_cust(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
                                show_bbox=False, captions=False, show_mask=False)
        tiempo_t=tiempo1+tiempo2+ tiempo3
        tiempo_total=str(tiempo_t)
        st.success("Tiempo de prediccion: "+tiempo_total)




        #display_instances_cust(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
                               #show_bbox=False, captions=False, show_mask=False)
      else:
         st.error('⛔️'+" ¡Por favor!, cargar imagen")

    if st.sidebar.button('📌️'+' About'):
        st.header("Video de funcionamiento del predictor de neumonia")
        video_file = open('ABOUT.mp4', 'rb',)
        video_bytes = video_file.read()
        st.video(video_bytes,start_time=0)
        st.markdown("Design by: Victor Manuel Astudillo Delgado")
        st.markdown("Estudiante de Ingenieria Mecatronica")


if __name__ == '__main__':
    main()
