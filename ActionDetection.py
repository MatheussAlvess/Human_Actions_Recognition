import os
import sys
import cv2 
import numpy as np
import pandas as pd
import mediapipe as mp 
import tensorflow as tf

# Definindo alias para facilitar o uso das funcionalidades do Mediapipe
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 

# Tamanhos predefinidos para landmarks de pose e mãos
pose_landmarks_size = 33
right_hand_landmarks_size = 21
left_hand_landmarks_size = 21

# Número total de coordenadas (pose + mãos)
num_coords = pose_landmarks_size + right_hand_landmarks_size + left_hand_landmarks_size


class ACTIONS():
    """
    Faz a predição frame por frame e printa na tela o nome da classe identificada a partir da localização da pose detectada.

    Args:
        model_path (str): Caminho do arquivo do modelo.
        dataset_path (str): Caminho do arquivo de coordenadas das classes.
        video_path (str): Diretório de vídeos.
        video_name (str): Nome do vídeo.
        real_time (bool): Define se a detecção será em tempo real (True) ou não (False).
        cam_device (int): Número do dispositivo de câmera, usado apenas se 'real_time' for True.
    Return: None.
    """

   
    def __init__(self,
                 model_path: object = 'mlp_model.h5',
                 dataset_path: object = 'dataset/coords.csv',
                 video_path: object = 'data',
                 video_name: object = 'demo.mp4',
                 real_time: bool = False,
                 cam_device: int = 0,
                 ) -> None:
        
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.video_path = video_path
        self.video_name = video_name
        self.real_time = real_time
        self.cam_device = cam_device

    # Método para carregar o modelo
    def load_model(self,model_path = None):
        model_path = self.model_path
        self.model = tf.keras.models.load_model(model_path)
        return self.model

    # Método para mapear as classes
    def class_mapping(self,dataset_path = None) -> None:
        dataset_path = self.dataset_path 
        self.df = pd.read_csv(dataset_path)
        self.df['label'] = self.df['label'].apply(lambda x: x.replace('1',''))
        self.class_map = {class_label: idx for idx, class_label in enumerate(self.df['label'].unique())}

    # Método para detecção de ações e predição de classe
    def actions_detection(self):
        try:
            if self.real_time:
                cap = cv2.VideoCapture(self.cam_device)  # Captura de vídeo em tempo real
            else:
                if os.path.exists(f'{self.video_path}/{self.video_name}'):
                    cap = cv2.VideoCapture(f'{self.video_path}/{self.video_name}') # Captura de vídeo gravado
                else:
                    print('Erro: não foi possível encontrar o video.')
                    
            # Inicializando Holistic com configurações de confiança mínima para detecçao e rastreamento
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                
                while cap.isOpened():
                    ret, frame = cap.read() # Leitura de cada frame do vídeo
                    
                    # Converte a imagem para o formato RGB (necessário para o MediaPipe)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Define a imagem como não-gravável para processamento interno do MediaPipe
                    image.flags.writeable = False        
                    
                    # Faz a detecção de pose, mãos e face no frame
                    results = holistic.process(image)
                    
                     # Define a imagem como gravável novamente para renderização
                    image.flags.writeable = True   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Desenha os landmarks da mão direita 
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                            )

                    # Desenha os landmarks da mão esquerda 
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            )

                    # Desenha os landmarks da pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )
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
                            r_h_row = [0,0,0,0] * right_hand_landmarks_size
                                
                        # Extrai os landmarks da mão esquerda
                        if results.left_hand_landmarks:
                            left_hand = results.left_hand_landmarks.landmark
                            l_h_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
                        else:
                            # Em casos da mão não ser detectada, preenche com zeros
                            l_h_row = [0,0,0,0] * right_hand_landmarks_size
                            
                         # Concatena as linhas de coordenadas em uma única linha
                        row = pose_row+r_h_row+l_h_row

                        # X recebe as coordenadas que serão utilizadas para predição
                        X = pd.DataFrame([row])

                        # Prediz de qual classe classe é a ação
                        index = np.argmax((self.model.predict(X)))
                        # Probabilidade da classe
                        class_prob = round(self.model.predict(X)[0][index],2)
                        # Mapeia o nome da classe de volta para string
                        body_language_class = list(self.class_map.keys())[index]
                            
                        # Pegando as coordenadas da orelha esquerda como referencia
                        coords = tuple(np.multiply(
                                        np.array(
                                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                    , [640,480]).astype(int))
                        
                        # Escrevendo na tela um retângulo
                        cv2.rectangle(image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                    (33, 107, 327), -1)
                        # Escrevendo na tela a classe predita
                        cv2.putText(image, body_language_class, coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Desenhando retangulo no canto superior esquerdo da tela
                        cv2.rectangle(image, (0,0), (250, 60), (33, 107, 237), -1)
                        
                        # Escrevendo classe predita
                        cv2.putText(image, 'CLASS'
                                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split(' ')[0]
                                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Escrevendo probabilidade da classe predita
                        cv2.putText(image, 'PROB'
                                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(class_prob)
                                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                    except:
                        pass

                    # Exibe o frame com as detecções                  
                    cv2.imshow('Video', image)
                    
                    # Aperte 'q' para sair
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()
        
        except:
            pass


if __name__=='__main__':
    args = sys.argv
   
    if len(args) > 1:
        human_actions = ACTIONS(real_time=True)
    else:
        human_actions = ACTIONS(video_name='libras.mp4')

    human_actions.load_model()
    human_actions.class_mapping()
    human_actions.actions_detection()