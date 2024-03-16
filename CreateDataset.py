import os
import csv
import cv2 
import numpy as np
import mediapipe as mp 


# Importa utilitários de desenho do MediaPipe
mp_drawing = mp.solutions.drawing_utils
# Importa o modelo holístico do MediaPipe (reconhece pose, mãos e face)
mp_holistic = mp.solutions.holistic

# Define o número de landmarks para cada parte do corpo
pose_landmarks_size = 33
right_hand_landmarks_size = 21
left_hand_landmarks_size = 21

# Calcula o número total de coordenadas (incluindo x, y, z e visibilidade)
num_coords = pose_landmarks_size + right_hand_landmarks_size + left_hand_landmarks_size

# Cria lista de nomes das colunas do CSV: label, seguido de x, y, z e v para cada landmark
landmarks = ['label']
for val in range(1, num_coords + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


def create_dataset(path='data', output_name='coords.csv', show=True):
    """
    Cria um dataset de coordenadas de pose, mãos e face a partir de vídeos.

    Args:
        path (str, optional): Caminho para o diretório contendo os vídeos. Default = 'data'.
                              É importante que os videos sejam nomeados com a ação representada, 
                              pois esse será o nome da classe.
        output_name (str, optional): Nome do arquivo CSV de saída. Default = 'coords.csv'.
        show (bool, optional): Se verdadeiro, exibe o vídeo com as detecções durante a criação do dataset. Default = True.
    """

    # Definindo diretório de destino
    output_path = f'dataset/{output_name}'

    try:
        os.mkdir(output_path.replace(output_name,''))
    except:
        print(f'Folder {output_path.replace(output_name,'')} already exists')
        

    with open(output_path, mode='w', newline='') as f:
        # Cria o escritor CSV para o arquivo de saída
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Escreve a linha do cabeçalho com os nomes das colunas
        csv_writer.writerow(landmarks)

    for file_name in os.listdir(path):
        # Monta o caminho completo do vídeo
        video_path = f'{path}/{file_name}'
        # Extrai a classe do vídeo a partir do nome do arquivo (supondo que a classe esteja na pasta do vídeo)
        class_name = file_name.split('/')[-1].split('.')[0]

        # Abre o vídeo usando OpenCV
        cap = cv2.VideoCapture(video_path)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    # Converte a imagem para o formato RGB (necessário para o MediaPipe)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Define a imagem como não-gravável para processamento interno do MediaPipe
                    image.flags.writeable = False

                    # Faz a detecção de pose, mãos e face no frame
                    results = holistic.process(image)

                    # Define a imagem como gravável novamente para renderização
                    image.flags.writeable = True
                    # Converte a imagem de volta para o formato BGR (necessário para OpenCV)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Desenha os landmarks da mão direita 
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

                    # Desenha os landmarks da mão esquerda 
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            )

                    # Desenha os landmarks da pose (magenta)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                    # Exporta as coordenadas dos landmarks
                    try:
                        # Extrai os landmarks da pose
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                        # Extrai os landmarks da mão direita
                        if results.right_hand_landmarks:
                            right_hand = results.right_hand_landmarks.landmark
                            r_h_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
                        else:
                            # Em casos da mão não ser detectada, preenche com zeros
                            r_h_row = [0, 0, 0, 0] * right_hand_landmarks_size

                        # Extrai os landmarks da mão esquerda
                        if results.left_hand_landmarks:
                            left_hand = results.left_hand_landmarks.landmark
                            l_h_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
                        else:
                            # Em casos da mão não ser detectada, preenche com zeros
                            l_h_row = [0, 0, 0, 0] * right_hand_landmarks_size

                        # Concatena as linhas de coordenadas em uma única linha
                        row = pose_row + r_h_row + l_h_row

                        # Insere a classe no início da linha
                        row.insert(0, class_name)

                        # Exporta a linha para o arquivo CSV
                        with open(output_path, mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)

                    except:
                        pass

                    if show:    
                        # Exibe o frame com as detecções         
                        cv2.imshow('Video', image)
                    
                    # Aperte 'q' para sair
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                else:
                    cap.release()
                    cv2.destroyAllWindows()


if __name__=='__main__':
    create_dataset()