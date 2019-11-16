# Projeto Reconhecimento de Audios
### Materia de Mineracao de Dados UFABC 

 * Para este projeto e necessario que as pastas com os dados de treino e validacao estejam nas pastas 'TREINAMENTO/' e 'VALIDACAO/', respectivamente, dentro da pasta principal do projeto.
 * Os dados devem estar formato .wav e duracao maxima de ~ 8s.

 ### Bibliotecas utilizadas
* librosa
* math
* glob
* random
* pandas
* numpy 
* matplotlib
* scipy
* re
* sklearn

### Como Executar
* Apos configurar as pastas com os dados para executar o projeto, basta  realizar o comando:
```
     python project.py
```
* Tomar o devido cuidado para nao executar utilizando python 2.

* Os testes foram realizados tendo o Python 3.6 como versão default do sistema.

### Resultados iniciais

* Para o classificador SVC (kernel linear) obtivemos uma taxa de acerto de aproximadamente 60%.

* Para o classificador KNN (k neighbours) obtivemos uma taxa de acerto de aproximadamente 57%.
