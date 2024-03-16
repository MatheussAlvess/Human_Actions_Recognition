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

Caso haja o interesse em executar os códigos e replicar os outputs, basta executar `$python vae_model.py`. Dessa forma, serão salvos tanto o encoder quanto o decoder com os parâmetros _default_ definidos na função `build_model`.

### Para executar os códigos e replicar os resultados, siga estas etapas:

1. Clone este repositório para o seu computador.
2. Navegue até o diretório do projeto.
3. Garanta ter as dependências necessárias (vide `requirements.txt`)
4. Descompacte a pasta do dataset e renomeie para 'data'. (Está pasta deve conter a subpasta _train_)
5. Execute o seguinte comando no terminal:
   `python vae_model.py`
   
#### O que o comando faz?

- O comando executará o script `vae_model.py`.
- Este script carrega os dados, treina o modelo VAE e salva os modelos encoder e decoder.
- Os modelos são salvos no diretório atual com os parâmetros _default_ definidos na função `build_model`.

#### Observações:

- Você pode modificar os parâmetros do modelo VAE editando o script `vae_model.py`, tanto passando os parâmetros para o `build_model` quanto variando internamente os hiperparâmetros da arquitetura (como o tamanho dos filtros, quantidade de camadas, etc.).
- A arquitetura foi fundamentada no [ _code examples_](https://keras.io/examples/generative/vae/) do Keras utilizando o MNIST, visando ter um ponto de partida "validado", uma vez que não seria possível realizar um maior refinamento ou até mesmo um grid de parâmetros e arquiteturas.
- Os resultados foram até aceitáveis, dado que nada foi otimizado apenas baseado. Mas para melhorar o desempenho, recomento aumentar o número de épocas, avaliar as métricas de perda, variar a estrutura tanto do encoder quanto do decoder além de variar os parâmetros no geral. 


**Nota**: Devido o tamanho dos modelos salvos, não foi possível subir no repositório. Para replicar os resultados, basta executar com os parâmetros _default_.
<img src="images/paz_tchau.gif" width="500" height="500"/>
