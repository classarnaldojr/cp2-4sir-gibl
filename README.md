# Attention Monitor

Sistema de monitoramento de atenção em tempo real usando visão computacional. Analisa o feed de uma webcam para detectar se o usuário está atento, distraído, sonolento ou ausente — gerando um score contínuo de atenção ao longo da sessão.

**Caso de uso primário:** cursos online, treinamentos corporativos e avaliações presenciais onde é necessário medir o engajamento sem hardware especializado.

---
## Integrantes
* Gustavo Carvalho - RM 550983
* Isaac Maranhos - RM 97847
* Bruno Granado - RM 551411
* Leticia Rocha - RM 552481
---

## Como funciona

O sistema combina dois sinais extraídos do rosto via MediaPipe FaceMesh:

### 1. Eye Aspect Ratio (EAR)

Métrica geométrica proposta por Soukupova & Cech (2016) que mede a abertura dos olhos a partir de seis pontos de landmark:

```
EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
```

Quando os olhos estão abertos, a distância vertical é alta em relação à horizontal. Quando fechados, o numerador colapsa próximo de zero. O limiar padrão é **0.25**: abaixo disso, os olhos são considerados fechados.

Se os olhos permanecerem fechados por 20 frames consecutivos (~0.67 s a 30 fps), o sistema classifica o usuário como **SONOLENTO**.

### 2. Estimativa de pose da cabeça

A direção do olhar é inferida pelo deslocamento do nariz em relação ao centro geométrico da face (ponto médio entre maçãs do rosto e entre testa e queixo). O resultado é normalizado pela largura e altura do rosto, tornando-o independente de resolução e distância da câmera.

Desvios acima de **0.12** (horizontal) ou **0.07** (vertical) classificam o usuário como **DISTRAÍDO**.

### 3. Score de atenção

Um score contínuo de 0 a 100 que decai a cada frame fora do estado ATENTO e se recupera quando a atenção é retomada. O decaimento é intencionalmente mais rápido que a recuperação — um desvio de alguns segundos deve ser perceptível no score, não compensado instantaneamente.

| Estado     | Variação por frame |
|------------|-------------------|
| ATENTO     | +0.15             |
| DISTRAIDO  | -0.30             |
| SONOLENTO  | -0.50             |
| AUSENTE    | -0.70             |

---

## Diferenciais em relação ao padrão

A maioria dos sistemas de monitoramento de atenção existentes exige:
- **Hardware de eye-tracking** dedicado (Tobii, Pupil Labs): custo entre R$ 3.000 e R$ 20.000 por unidade.
- **Infraestrutura de cloud** para inferência com modelos de deep learning.
- **Câmeras infravermelhas** para funcionamento em ambientes com iluminação variável.

Este sistema opera com:
- Qualquer webcam USB ou integrada (já presente em notebooks).
- Processamento **100% local** — sem envio de imagem para servidores.
- Análise **geométrica** (EAR + pose), sem modelos de redes neurais para classificação, o que garante latência baixa e funcionamento offline.

A limitação consciente da abordagem é a sensibilidade à iluminação frontal precária e ao uso de óculos. Ambientes bem iluminados resolvem o primeiro caso; o segundo é uma troca aceita em favor da simplicidade e custo zero.

---

## Contexto: São Paulo e o problema de engajamento em aprendizado

São Paulo concentra a maior densidade de instituições de ensino superior e centros de treinamento corporativo da América do Sul. Com o crescimento do ensino híbrido e a massificação de plataformas LMS pós-pandemia, um problema persistente emergiu: **presença não é sinônimo de atenção**.

Dados do setor educacional brasileiro indicam que mais de 40% dos alunos em aulas remotas realizam outras atividades em paralelo durante o horário de aula. Do lado corporativo, empresas de médio e grande porte em São Paulo gastam em média R$ 1.200 por colaborador por ano em treinamentos — com taxas de retenção frequentemente abaixo de 30%.

**Ganhos potenciais desta implementação:**

- **Feedback em tempo real para o instrutor:** em uma sala com múltiplos participantes visíveis, o score médio pode sinalizar queda de engajamento antes que o conteúdo seja perdido, permitindo intervenções — pausas, dinâmicas, perguntas direcionadas.
- **Dados para redesenho de conteúdo:** um log de atenção ao longo de uma aula identifica os momentos de queda sistemática, apontando onde o material precisa de revisão.
- **Triagem em avaliações:** detectar padrões de desvio de olhar durante provas é uma aplicação direta de proctoring, substituindo soluções pagas como ProctorU ou Respondus Monitor.
- **Custo de implantação próximo de zero:** funciona no hardware que instituições e empresas já possuem, sem licenciamento adicional.

---

## Arquitetura

```
attention-monitor/
├── src/
│   ├── detection/
│   │   ├── eye_analyzer.py        # Calculo de EAR — sem dependencias pesadas
│   │   ├── head_pose_analyzer.py  # Estimativa de pose — sem dependencias pesadas
│   │   ├── types.py               # DetectionResult (dataclass compartilhado)
│   │   └── face_detector.py       # Integracao com MediaPipe FaceMesh
│   ├── scoring/
│   │   └── attention_scorer.py    # Maquina de estados + score de sessao
│   ├── ui/
│   │   └── dashboard.py           # Overlay OpenCV + painel lateral
│   └── reporting/
│       └── session_report.py      # Relatorio de fim de sessao
├── tests/
│   ├── test_eye_analyzer.py
│   ├── test_head_pose_analyzer.py
│   └── test_attention_scorer.py
├── main.py
├── requirements.txt
└── pyproject.toml
```

A separacao entre `eye_analyzer.py` / `head_pose_analyzer.py` e `face_detector.py` e deliberada: os dois primeiros sao modulos de calculo puro (apenas numpy), diretamente testáveis sem webcam ou MediaPipe. `face_detector.py` e o unico modulo com dependencias de I/O — isolar essa fronteira facilita testes e troca futura de modelo de deteccao facial.

---

## Instalacao

```bash
pip install -r requirements.txt
```

Python 3.9+ recomendado. Testado com Python 3.10 e 3.11.

---

## Uso

```bash
# Camera padrao (indice 0)
python main.py

# Camera especifica
python main.py 1
```

**Controles durante a sessao:**

| Tecla | Acao                        |
|-------|-----------------------------|
| Q     | Encerrar e exibir relatorio |
| ESC   | Encerrar e exibir relatorio |
| R     | Reiniciar score da sessao   |

Ao fechar, o relatorio de sessao e impresso no terminal com a distribuicao de estados e o score final.

---

## Testes

Os testes validam os algoritmos de forma deterministica — sem webcam, sem MediaPipe, sem frames reais. Cada funcao de calculo recebe entradas sinteticas com resultado esperado conhecido.

```bash
pytest
```

**Cobertura dos testes:**

- `test_eye_analyzer.py`: valores conhecidos de EAR, proporcionalidade, invariancia a escala, limiar customizavel, casos degenerados (olho completamente fechado, span horizontal zero).
- `test_head_pose_analyzer.py`: deteccao de yaw positivo/negativo, pitch positivo/negativo, caso frontal, casos degenerados (face sem dimensao), valor conhecido de yaw.
- `test_attention_scorer.py`: estado inicial, dinamica de score (cap em 100, floor em 0, decaimento vs recuperacao), transicoes de estado (DISTRAIDO, SONOLENTO, AUSENTE com limiares de frame), contagem exclusiva de frames, calculo de porcentagem de atencao.

---

## Referencias

- Soukupova, T., & Cech, J. (2016). *Real-time eye blink detection using facial landmarks*. 21st Computer Vision Winter Workshop.
- Lugaresi, C., et al. (2019). *MediaPipe: A framework for building perception pipelines*. arXiv:1906.08172.
- Bradski, G. (2000). *The OpenCV library*. Dr. Dobb's Journal of Software Tools.
