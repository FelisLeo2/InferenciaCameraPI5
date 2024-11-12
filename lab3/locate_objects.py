import argparse
import datetime
import io
import shutil
import time
import os
import base64
import json
from queue import Queue
import threading
import requests

import numpy as np
from PIL import Image, ImageDraw
import tflite_runtime.interpreter as tflite

import cv2
from nms import non_max_suppression_yolov8

# arquivos para debug
OUTPUT_DIR = "imagem_rosto"
OUTPUT_FILE = "data_rosto.json"

# parâmetros globais
CAM_HIGH_RESOLUTION = (2592, 1944)
CAM_LOW_RESOLUTION = (320, 240)
THRESHOLD = 0.7
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
ROI_DEFINED = False
ROI_COORDS = (0, 0, 222, 200)

<<<<<<< HEAD
# Endereço do servidor
ADDRESS = 'http://192.168.0.87:8000/'

# fila com as matrizes do rosto
face_data_queue_ROI = Queue()
face_data_queue_hands = Queue()
face_data_aux = 0


# keypoints gerados pelo YOLO
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
LEFT_EAR_INDEX = 3
RIGHT_EAR_INDEX = 4
LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
LEFT_FOOT_INDEX = 15
RIGHT_FOOT_INDEX = 16

REQUIRED_KEYPOINTS = [
    LEFT_EYE_INDEX, RIGHT_EYE_INDEX,
    LEFT_EAR_INDEX, RIGHT_EAR_INDEX,
    LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX,
    # LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
]

# Quantas coordenadas estão presentes na box
=======
cam_high_resolution = (2592, 1944)
cam_low_resolution = (320, 240)
threshold = 0.7
keypoint_confidence_threshold = 0.5

roi_defined = False
roi_coords = (0,0,0,0)

# fila com as matrizes do rosto
face_data_queue_ROI = Queue()
face_data_queue_hands = Queue()

output_file = "data_rosto.json"

left_eye_index = 1
right_eye_index = 2
left_ear_index = 3
right_ear_index = 4
left_shoulder_index = 5
right_shoulder_index = 6
left_foot_index = 15
right_foot_index = 16 

required_keypoints = [
    left_eye_index, right_eye_index,
    left_ear_index, right_ear_index,
    left_shoulder_index, right_shoulder_index,
    #left_foot_index, right_foot_index
]

def define_roi(frame):
    """
    Define a região de interesse (ROI) desenhando um retângulo vermelho no primeiro frame.
    """
    global roi_defined, roi_coords

    # Coordenadas do retângulo inicial (você pode ajustar conforme necessário)
    x, y, w, h = 20, 20, 600, 600  # exemplo de posição e tamanho

    # Desenha o retângulo no primeiro frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    roi_coords = (x, y, w, h)
    roi_defined = True

def save_head_region(imagem_high, keypoints, face_data_aux, timestamp, output_path, extra_pixels=0):
  left_eye_x = keypoints[6 + left_eye_index * 3]
  left_eye_y = keypoints[6 + left_eye_index * 3 + 1]
  right_eye_x = keypoints[6 + right_eye_index * 3]
  right_eye_y = keypoints[6 + right_eye_index * 3 + 1]
  left_ear_x = keypoints[6 + left_ear_index * 3]
  right_ear_x = keypoints[6 + right_ear_index * 3]
  left_shoulder_y = keypoints[6 + left_shoulder_index * 3 + 1]
  right_shoulder_y = keypoints[6 + right_shoulder_index * 3 + 1]

    # Calcular o centro do rosto
  center_x = (left_eye_x + right_eye_x) / 2
  center_y = (left_eye_y + right_eye_y) / 2

  # Calcular as coordenadas da região do rosto
  max_ear_lenght_x = max(abs(left_ear_x - center_x), abs(right_ear_x - center_x))
  max_shoulder_lenght_y = max(abs(left_shoulder_y - center_y), abs(right_shoulder_y - center_y))

  max_x = center_x + max_ear_lenght_x + extra_pixels
  max_y = center_y + max_shoulder_lenght_y 
  min_x = center_x - max_ear_lenght_x - extra_pixels
  min_y = center_y - max_shoulder_lenght_y
  
  min_x_high = min_x #* scale_x
  min_y_high = min_y #* scale_y
  max_x_high = max_x #* scale_x
  max_y_high = max_y #* scale_y

  ##################################################################################
  
  ##################################################################################

  if imagem_high.mode == 'RGBA':
    imagem_high = imagem_high.convert('RGB')

  head_region = imagem_high.crop((min_x_high, min_y_high, max_x_high, max_y_high))

  head_region.save(output_path)

  # Converter para base64
  buffered = io.BytesIO()
  head_region.save(buffered, format="JPEG")
  img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8") 

  # Dados do rosto a serem salvos
  face_data = {
      "time": timestamp,
      "base64_image": img_base64
  }

  face_data2 = {
      "qtd": face_data_aux.qsize()
  }

  face_data_aux.put(face_data)

  # Carregar dados existentes do arquivo (se houver)
  try:
      with open(output_file, "r") as infile:
          all_faces = json.load(infile)
  except FileNotFoundError:
      # Se o arquivo não existir, inicializar uma lista vazia
      all_faces = []

  # Adicionar o novo rosto à lista
  all_faces.append(face_data2)

  # Salvar os dados atualizados de volta no arquivo
  with open(output_file, "w") as outfile:
      json.dump(all_faces, outfile, indent=4)

# How many coordinates are present for each box.
>>>>>>> main
BOX_COORD_NUM = 4

# Como desenhar o esqueleto a partir dos keypoints 
POSE_LINES = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [
    5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [13, 11], [14, 12], [15, 13], [16, 14]]

def roi_drawn(frame):
  """
  Define a região de interesse (ROI) desenhando um retângulo vermelho no primeiro frame.
  """
  global ROI_DEFINED, ROI_COORDS

  # Desenha o retângulo no primeiro frame
  (x, y, w, h) = ROI_COORDS
  cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
  ROI_DEFINED = True

def save_head_region(imagem_high, keypoints, timestamp, output_path, extra_pixels=0):
  """
  Recorta o rosto da imagem capturada 
  """
  global face_data_aux
  # Obter keypoints para o recorte do rosto
  left_eye_x = keypoints[6 + LEFT_EYE_INDEX * 3]
  left_eye_y = keypoints[6 + LEFT_EYE_INDEX * 3 + 1]
  right_eye_x = keypoints[6 + RIGHT_EYE_INDEX * 3]
  right_eye_y = keypoints[6 + RIGHT_EYE_INDEX * 3 + 1]
  left_ear_x = keypoints[6 + LEFT_EAR_INDEX * 3]
  right_ear_x = keypoints[6 + RIGHT_EAR_INDEX * 3]
  left_shoulder_y = keypoints[6 + LEFT_SHOULDER_INDEX * 3 + 1]
  right_shoulder_y = keypoints[6 + RIGHT_SHOULDER_INDEX* 3 + 1]

  # Calcular o centro do rosto
  center_x = (left_eye_x + right_eye_x) / 2
  center_y = (left_eye_y + right_eye_y) / 2

  # Calcular as coordenadas da região do rosto
  max_ear_lenght_x = max(abs(left_ear_x - center_x), abs(right_ear_x - center_x))
  max_shoulder_lenght_y = max(abs(left_shoulder_y - center_y), abs(right_shoulder_y - center_y))

  max_x = center_x + max_ear_lenght_x + extra_pixels
  max_y = center_y + max_shoulder_lenght_y 
  min_x = center_x - max_ear_lenght_x - extra_pixels
  min_y = center_y - max_shoulder_lenght_y
  
  min_x_high = min_x #* scale_x
  min_y_high = min_y #* scale_y
  max_x_high = max_x #* scale_x
  max_y_high = max_y #* scale_y

  if imagem_high.mode == 'RGBA':
    imagem_high = imagem_high.convert('RGB')

  head_region = imagem_high.crop((min_x_high, min_y_high, max_x_high, max_y_high))

  # Converter para base64
  buffered = io.BytesIO()
  img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8") 

  # Dados do rosto a serem salvos
  face_data = {
      "time": timestamp,
      "base64_image": img_base64
  }
  # Debug de números de faces salvas
  face_data_aux = face_data_aux + 1
  face_data2 = {
      "qtd": face_data_aux
  }

  # Carregar dados existentes do arquivo (se houver)
  try:
      with open(output_file, "r") as infile:
          all_faces = json.load(infile)
  except FileNotFoundError:
      # Se o arquivo não existir, inicializar uma lista vazia
      all_faces = []

  # Adicionar o novo rosto à lista
  all_faces.append(face_data2)

  # Salvar os dados atualizados de volta no arquivo
  with open(output_file, "w") as outfile:
      json.dump(all_faces, outfile, indent=4)
  return face_data

def load_labels(filename):
  """
  Método para carregar rótulos de um arquivo
  """
  with open(filename, "r") as f:
    return [line.strip() for line in f.readlines()]

def faceDetect():
  """
  Método de detecção de rostos
  """
  global face_data_queue_ROI, face_data_queue_hands

<<<<<<< HEAD
  fps_start_time = 0
  fps = 0

  # Captura de vídeo da câmera
=======
def faceDetect():
>>>>>>> main
  if args.camera is not None:
    ##################################################################################
    cap = cv2.VideoCapture(args.camera)  # Usando OpenCV para capturar da câmera USB
    ##################################################################################
<<<<<<< HEAD

  # Carregar o modelo TFLite
=======
>>>>>>> main
  interpreter = tflite.Interpreter(
      model_path='../models/yolov8n-pose_int8.tflite',
      num_threads=2)
  interpreter.allocate_tensors()

  class_labels = load_labels('../models/yolov8-pose_labels.txt')

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Verifica o tipo do tensor de entrada
  floating_model = input_details[0]["dtype"] == np.float32

  # NxHxWxC, H:1, W:2
  input_height = input_details[0]["shape"][1]
  input_width = input_details[0]["shape"][2]
  max_box_count = output_details[0]["shape"][2]

  # Número de caixas e pontos chave
  class_count = 1
  box_num_values = output_details[0]["shape"][1]
  non_keypoint_values = (BOX_COORD_NUM + 1)
  keypoint_values = box_num_values - non_keypoint_values
  if (keypoint_values % 3) != 0:
    print(
        f"Inesperado número de valores {keypoint_values} para o formato yolov8_pose")
    exit(0)
  keypoint_count = int(keypoint_values / 3)

  if len(class_labels) != class_count:
    print("Modelo tem %d classes, mas %d rótulos" %
          (class_count, len(class_labels)))
    exit(0)

  while True:
    ##################################################################################
    # Captura de um quadro da câmera
    ret, frame = cap.read()

    # Cálculo da taxa de quadros por segundo (FPS)
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/(time_diff)
    fps_start_time = fps_end_time
    # print(fps)
    frame = cv2.resize(frame, (input_width, input_height))
    timestamp = time.time()
    if not ret:
      print("Falha ao capturar quadro")
      break
    # Definindo a região de interesse (ROI) se não estiver definida
    if not ROI_DEFINED:
        define_roi(frame)
    else:
<<<<<<< HEAD
        # Desenha o retângulo nos frames subsequentes
        a, b, c, d = ROI_COORDS
        cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 0, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    ##################################################################################
    # Calculando a escala da imagem com menos e mais qualidade
    scale_x = 1#2592 / input_width
    scale_y = 1#1944 / input_height
=======
      ##################################################################################
      ret, frame = cap.read()
      timestamp = time.time()
      if not ret:
        print("Failed to grab frame")
        break
      
      if not roi_defined:
          define_roi(frame)
      else:
          # Desenha o retângulo nos frames subsequentes
          a, b, c, d = roi_coords
          cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 0, 255), 2)

      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(frame_rgb).resize((input_width, input_height), resample=Image.Resampling.LANCZOS)
      ##################################################################################

    scale_x = 1#2592 / input_width
    scale_y = 1#1944 / input_height
    
    if args.save_input is not None:
      img.save("new_" + args.save_input)
      # Do a file move to reduce flicker in VS Code.
      shutil.move("new_" + args.save_input, args.save_input)
>>>>>>> main

    # Redimensiona a imagem para se adequar à entrada do modelo
    input_data = np.expand_dims(img, axis=0)
    if floating_model:
      input_data = (np.float32(input_data) - 0.0) / 255.0

    interpreter.set_tensor(input_details[0]["index"], input_data)
    start_time = time.time()

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    results = np.squeeze(output_data).transpose()

    # Processa as caixas detectadas
    boxes = []
    for i in range(max_box_count):
      raw_box = results[i]
      center_x = raw_box[0]
      center_y = raw_box[1]
      w = raw_box[2]
      h = raw_box[3]

      score = raw_box[BOX_COORD_NUM]
      if (score > 0.6):
        coords = [center_x, center_y, w, h, score, 0]
        keypoints = raw_box[BOX_COORD_NUM + 1:]
        boxes.append([*coords, *keypoints])

    # Limpa as caixas sobrepostas. Veja
    # https://petewarden.com/2022/02/21/non-max-suppressions-how-do-they-work/
    clean_boxes = non_max_suppression_yolov8(
        boxes, class_count, keypoint_count)

    if args.save_output is not None:
      img_draw = ImageDraw.Draw(img)

    for box in clean_boxes:
<<<<<<< HEAD
      # Informação dos keypoints que serão usados para recorte do rosto
      keypoint_references = [box[6 + i * 3] for i in REQUIRED_KEYPOINTS]
      keypoint_confidences = [box[8 + i * 3] for i in REQUIRED_KEYPOINTS]

      # Salva as coordenadas de cada bounding box capturada na imagem.
=======
      keypoint_references = [box[6 + i * 3] for i in required_keypoints]
      keypoint_confidences = [box[8 + i * 3] for i in required_keypoints]
      # print("Box: ", box)
      # Convert from yolov8 coords to OpenCV coords.
>>>>>>> main
      center_x = box[0] * input_width
      center_y = box[1] * input_height
      w = box[2] * input_width
      h = box[3] * input_height
      half_w = w / 2
      half_h = h / 2
      left_x = int(center_x - half_w)
      right_x = int(center_x + half_w)
      top_y = int(center_y - half_h)
      bottom_y = int(center_y + half_h)

      # Informações sobre as bounding boxes
      score = box[4]
      class_index = box[5]
      class_label = class_labels[class_index]

      right_hand = []
      left_hand = []
<<<<<<< HEAD
   
      # Desenhar informações das bounding boxes na imagem
=======
      #print(f"{class_label}: {score:.2f} ({center_x:.0f}, {center_y:.0f}) {w:.0f}x{h:.0f}")
>>>>>>> main
      if args.save_output is not None:
        img_draw.rectangle(((left_x, top_y), (right_x, bottom_y)), fill=None)
        for line in POSE_LINES:
          start_index = 6 + (line[0] * 3)
          end_index = 6 + (line[1] * 3)
          start_x = box[start_index + 0]
          start_y = box[start_index + 1]
          end_x = box[end_index + 0]
          end_y = box[end_index + 1]
          img_draw.line((start_x, start_y, end_x, end_y), fill="yellow")
        for i in range(keypoint_count):
          index = 6 + (i * 3)
          k_x = box[index + 0]
          k_y = box[index + 1]
          img_draw.arc((k_x - 1, k_y - 1, k_x + 1, k_y + 1), start=0, end=360, fill="red")

          # Obter os keypoints das mãos
          if (i == 10):
            right_hand = [k_x, k_y]
          if (i == 9):
            left_hand = [k_x, k_y]

<<<<<<< HEAD
        # Implementação da lógica de mãos juntas 
        hands_together = ((abs(right_hand[0] - left_hand[0])/w)<0.4) and ((abs(right_hand[1] - left_hand[1])/h)<0.1)

      # Implementação do filtro de imagens borradas
      if score > THRESHOLD and all(keypoint_references) and all(conf >= KEYPOINT_CONFIDENCE_THRESHOLD for conf in keypoint_confidences):
        # print(box[6 + 3 * 3] - box[6 + 4 * 3])
=======
              # img_draw.text((k_x, k_y), f"([{i}] {k_x:.0f},{k_y:.0f})", fill=(0, 0, 0))
            # img_draw.text((k_x, k_y), f"([{i}] {k_x:.0f},{k_y:.0f})", fill=(0, 0, 0))
          '''
          print("REL_X: ", abs(right_hand[0] - left_hand[0])/w)
          print("REL_Y: ", abs(right_hand[1] - left_hand[1])/h)
          print("DIF_X: ", abs(right_hand[0] - left_hand[0]))
          print("DIF_Y: ", abs(right_hand[1] - left_hand[1]))
          print("W: ", w)
          '''
          hands_together = ((abs(right_hand[0] - left_hand[0])/w)<0.4) and ((abs(right_hand[1] - left_hand[1])/h)<0.1)
          '''
          if (w <= 38):
            print("Pessoa distante")
            img_draw.text((right_x-15, top_y), "FORA", fill=(255, 0, 0))
          elif (hands_together):
            print("Mãos juntas")
            img_draw.text((right_x-15, top_y), "MJ", fill=(0, 0, 0))
          else:
            print("Mãos separadas")
            img_draw.text((right_x-15, top_y), "MS", fill=(0, 0, 255))
            '''
      if score > threshold and all(keypoint_references) and all(conf >= keypoint_confidence_threshold for conf in keypoint_confidences):
        print(box[6 + 1 * 3] - box[6 + 2 * 3])
        if(left_x >= roi_coords[0] and top_y >= roi_coords[1] and right_x <= roi_coords[0] + roi_coords[2] and bottom_y <= roi_coords[1] + roi_coords[3]):
            save_head_region(img, box, face_data_queue_ROI, timestamp, f"imagem_rosto/head_{int(time.time() * 1000)}.jpg")
        if hands_together:
          save_head_region(img, box, face_data_queue_hands, timestamp, f"imagem_rosto/head_{int(time.time() * 1000)}.jpg")
>>>>>>> main

        # Implementação da lógica de dentro ou fora do leito
        # if((box[6 + 3 * 3] - box[6 + 4 * 3]) > 21):
        if(left_x < ROI_COORDS[0] or top_y < ROI_COORDS[1] or right_x > ROI_COORDS[0] + ROI_COORDS[2] or bottom_y > ROI_COORDS[1] + ROI_COORDS[3]) and ((box[6 + 3 * 3] - box[6 + 4 * 3]) > 21):
          img_draw.text((left_x, top_y), "FORA DO LEITO", fill=(120, 0, 255))
          # Método de recortar a face
          cutted_face = save_head_region(img, box, timestamp, f"imagem_rosto/head_{int(time.time() * 1000)}.jpg")

          # Salvar na fila de pessoas dentro do leito
          face_data_queue_ROI.put(cutted_face)

          if hands_together:
            # Salvar na fila de pessoa higienizadas
            face_data_queue_hands.put(cutted_face)
        
        else:
          img_draw.text((left_x, top_y), "DENTRO DO LEITO", fill=(0, 255, 0))

    # Debug de saída de imagem
    if args.save_output is not None:
      img.save("new_" + args.save_output)
      shutil.move("new_" + args.save_output, args.save_output)

<<<<<<< HEAD
    # Fecha a câmera
=======
    stop_time = time.time()
    #print("time: {:.3f}ms".format((stop_time - start_time) * 1000))

>>>>>>> main
    if args.camera is None:
      ##################################################################################
      cap.release()
      ##################################################################################

def sendFaceData_ROI():
<<<<<<< HEAD
  """
  Método para envio das imagens recortadas de pessoas dentro do leito
  """
  global ADDRESS, face_data_queue_ROI
  while True:
    if(face_data_queue_ROI.empty()):
      pass 

    try:
      # Retira o item da fila e o retorna
      item = face_data_queue_ROI.get()

      # Envia a foto via HTTP
      response = requests.post(ADDRESS + 'upload_ROI', json=item, timeout=5)

    except requests.exceptions.ConnectionError as e:
      print("Erro de conexão:", e)
    except requests.exceptions.RequestException as e:
      print(f"Erro ao enviar dados: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")
    

def sendFaceData_hands():
  """
  Método para envio das imagens recortadas de pessoas com as mãos Higienizadas
  """
  global ADDRESS, face_data_queue_hands
  while True:
    if(face_data_queue_hands.empty()):
      pass

    try:
      # Retira o item da fila e o retorna
      item = face_data_queue_hands.get() 

      # Envia a foto via HTTP
      response = requests.post(ADDRESS + 'upload_maos', json=item, timeout=5) 

    except requests.exceptions.ConnectionError as e:
      print("Erro de conexão:", e)
    except requests.exceptions.RequestException as e:
      print(f"Erro ao enviar dados: {e}")
    except Exception as e:
      print(f"Erro inesperado: {e}")
=======
  while True:
    try:
      item = face_data_queue_ROI.get()  # Use timeout para evitar bloqueio
      response = requests.post('http://192.168.0.87:8000/upload_ROI', json=item, timeout=5)  # Use json=item para enviar os dados como JSON
    except face_data_queue_ROI.empty():
      # Não faz nada se a fila estiver vazia, apenas aguarda
      continue
    except requests.exceptions.RequestException as e:
      print(f"Erro ao enviar dados: {e}")

def sendFaceData_hands():
  while True:
    try:
      item = face_data_queue_hands.get()  # Use timeout para evitar bloqueio
      response = requests.post('http://192.168.0.87:8000/upload_maos', json=item, timeout=5)  # Use json=item para enviar os dados como JSON
    except face_data_queue_hands.empty():
      # Não faz nada se a fila estiver vazia, apenas aguarda
      continue
    except requests.exceptions.RequestException as e:
      print(f"Erro ao enviar dados: {e}")
>>>>>>> main

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
<<<<<<< HEAD
      "--camera", default=None, type=int, help="Pi camera device to use")
  parser.add_argument(
      "--save_output", default=None, help="Image file to save model output to")

  args = parser.parse_args()

  # Criando threads dos 3 principais métodos do programa
=======
      "-i",
      "--image",
      default="../images/bus.jpg",
      help="Input image")
  parser.add_argument(
      "-m",
      "--model_file",
      default="../models/yolov8n_224_int8.tflite",
      help="TF Lite model to be executed")
  parser.add_argument(
      "-l",
      "--label_file",
      default="../models/yolov8_labels.txt",
      help="name of file containing labels")
  parser.add_argument(
      "--input_mean",
      default=0.0, type=float,
      help="input_mean")
  parser.add_argument(
      "--input_std",
      default=255.0, type=float,
      help="input standard deviation")
  parser.add_argument(
      "--num_threads", default=2, type=int, help="number of threads")
  parser.add_argument(
      "--camera", default=None, type=int, help="Pi camera device to use")
  parser.add_argument(
      "--save_input", default=None, help="Image file to save model input to")
  parser.add_argument(
      "--save_output", default=None, help="Image file to save model output to")
  parser.add_argument(
      "--score_threshold",
      default=0.6, type=float,
      help="Score level needed to include results")
  parser.add_argument(
      "--output_format",
      default="yolov8_detect",
      help="How to interpret the output from the model"
  )

  args = parser.parse_args()

>>>>>>> main
  thread_productor = threading.Thread(target=faceDetect)
  thread_consummer_ROI = threading.Thread(target=sendFaceData_ROI)
  thread_consummer_hands = threading.Thread(target=sendFaceData_hands)

  thread_productor.start()
  thread_consummer_ROI.start()
  thread_consummer_hands.start()
