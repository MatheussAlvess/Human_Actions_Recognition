# actions

# Reconhecimento de Ações Humanas em Vídeo
<img src="images/exemplo.gif" width="500" height="500"/>

## O projeto consiste na construção de um sistema que permite reconhecer 20 ações humanas diferentes. 

**As 20 ações são:**

```
1. Amigo (Sinal LIBRAS)
2. Carro (Sinal LIBRAS)
3. Chuveiro (Sinal LIBRAS)
4. Conhecer (Sinal LIBRAS)
5. Coração
6. Curso (Sinal LIBRAS)
7. Double Biceps
8. Esconder
9. Legal
10. LIBRAS (Sinal LIBRAS)
11. Moto (Sinal LIBRAS)
12. Obrigado (Sinal LIBRAS)
13. Paz
14. Relaxado
15. Sem ação
16. Sentido
17. Tchau
18. Telefone (Sinal LIBRAS)
19. Tomada (Sinal LIBRAS)
20. Trabalhar (Sinal LIBRAS)
```

## Resumo:
Foi utilizado o framework MediaPipe para realizar a detecção das poses humanas. Uma vez detectadas, as poses foram passadas de input para um modelo MLP (Multi Layer Perceptron) para que fosse realizada a classificação da ação.
Os dados de treinamento para a MLP foram geradas passando um vídeo e extraíndo as poses humanas do mesmo, quando extraídas frame by frame, foram salvas em um CSV (Comma Separated Values).

## Dataset:
  
- O dataset consiste em uma coluna `label`, referente às ações, e as demais colunas são as coordenadas dos landmarks detectados pelo MediaPipe, onde essas serão as _features_ do modelo de classificação.
- Ele foi criado a partir de vídeos salvos na pasta `data`. (adendo: a pasta `data` deste repositório não contém todos os vídeos considerados no projeto)  

## Conteúdo:

**No repositório podem ser encontrados os arquivos:**
- `CrateDataset.py` - > Neste arquivo, é criado o dataset a partir dos vídeos de interesse.
- `MLPModel.py` -> Neste arquivo, é construído e treinado o modelo MLP.
- `ActionDetection.py` -> Neste arquivo, é feita a leitura do vídeo (seja de um vídeo gravado ou em tempo real) e feita as detecções dos landmarks. Uma vez com os landmarks, é utilizado o modelo carregado para realizar as classificações das ações.
- `data` -> Está pasta contém os vídeos utilizados para a criação do dataset de coordenadas.
- `dataset` -> Está pasta contém o dataset das coordenadas referente aos vídeos da pasta `data`.
- `outputs` e `images` -> São pastas auxiliares para armazenar os resultados do projeto.

## Uso:

**Passos iniciais**
1. Clone este repositório para o seu computador.
   ```
   git clone https://github.com/MatheussAlvess/<nome_do_repositório>.git
   ```
3. Navegue até o diretório do projeto.
4. Garanta ter as dependências necessárias (vide `requirements.txt`)
   
- **Para realizar o reconhecimento de uma das 20 ações em um vídeo dado de input, execute o comando**

  ```
  python ActionDetection.py
  ```
  Garantindo que dentro do arquivo `ActionDetection.py` a classe "ACTIONS" seja instaciada com o nome do vídeo de interesse de input, assim o vídeo será processado e salvo com o nome `output_<nome_do_video>`. 
  
  > Ex.: Executando `python ActionDetection.py`, tendo estanciado "ACTIONS(video_name='libras.mp4')" dentro do arquivo, será salvo um vídeo nomeado `output_libras.mp4`.
  

- **Caso queira fazer o reconhecimento e classificação da ação em tempo real, basta executar:**

  ```
  python ActionDetection.py live
  ```
  
  Assim a webcam será aberta e poderá ser feito o reconhecimento em tempo real.
  
___________________________________________
  
## Para utilizar o projeto como base para um projeto próprio, realize as seguinte etapas:

**Uma vez que tudo esteja pronto para ser executado (repositório clonado):**

1. Armazene seus vídeos dentro de uma pasta, onde cada vídeo é nomeado de acordo com a ação, pois a classe é obtida a partir do nome do arquivo:
   Ex.: `data/libras.mp4`   
2. Execute o comando:
   ```
   python CreateDataset.py
   ```
   Dessa forma será criado o dataset de coordenadas a partir dos vídeos encontrados na pasta de referência. (Por _default_ é "data")
3. Execute o comando:
   ```
   python MLPModel.py
   ```
   Assim o modelo MLP será treinado com base no dataset de coordenadas. (A arquitetura e parâmetros podem ser modificados dentro do arquivo)
4. Por fim execute o comando para reconhecimento das ações:
   ```
   python ActionDetection.py
   ``` 

___________________________________________

#### Observações:

- Esta é a primeira versão do projeto, dessa forma, as classificações com base nas detecções podem não ser tão precisas para algumas ações.
  Isso se deve por alguns motivos, sendo alguns deles:
  1. Conjunto de dados relativamente pequeno: Considerei apenas um video curto para cada ação e as ações não variavam muito. Por exemplo, para aprender a ação 'paz'
     o modelo recebe um cenário onde uma mão está com 2 dedos levantados enquanto que a outra não está visível na imagem, logo, existe a associação de que quando um mão das mãos não está visível isso pode se configurar a ação 'paz',
     algo que não é necessariamente correto.
  
  2. Não houve um tratamento do dataset de coordenadas. Em alguns cenários a ação do sinal 'amigo' não tinha a detecção de nenhuma das mãos, o que faz com que o modelo entenda que a ausência das mão pode ser considerado a ação do sinal 'amigo'.
     
  3. Refinamento do modelo. O modelo MLP considerado não foi otimizado, em alguns cenários ele pode estar fazendo a associação dos valores das coordenadas com a ação que não necessariamente é a correta, como acontece para 'tchau', 'paz', 'telefone'.
     Por serem ações que tem as coordenadas muito parecidas, o modelo pode não ser robusto para identificar a classe.
     
  4. O ponto que, ao entendimento adiquirido durante a execução do projeto, mais impacta na confusão da classificação das ações é a falha de detecção dos landmarks.
     Como o modelo classificador depende das coordenadas, quando a detecção dos landmarks falham, o classificador fica perdido. E isso é mais grave no contexto de treinamento, pois o modelo pode estar aprendendo
     que a ausência de coordenadas pora as mãos é o sinal "telefone". Isso pode ser resolvido tatno com o tratamento dos dados, melhora na resolução do vídeo (facilitando a detectção) ou até mesmo considerar
     outro modelo para a detecção que seja mais eficiente.



> [!TIP]
> Trabalhando em um outro projeto com MediaPipe, já tive experiência com o problema da falha de detecção de landmarks. Como alternativa, utilizei o Pose Estimation da YOLO, a qual é bem mais eficiente realizando as detecções (em troca de um maior custo computacional). [Utilizando YOLO para Pose Estimation](https://github.com/MatheussAlvess/Cervical_Posture_YOLO_Pose_Estimation). 

<img src="images/paz_tchau.gif" width="500" height="500"/>
