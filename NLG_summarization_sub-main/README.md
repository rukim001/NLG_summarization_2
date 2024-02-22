# NLG_summarization
유저가 쉽게 정보를 파악할 수 있도록 뉴스의 핵심 추출  
핵심 문장이 잘 만들었는지 판단할 수 있는 NLG 모델 개발

## Process
Data -> Data Preprocessing -> Data Augumentation -> True Summary Generation → Model → Param Tuning -> Result

## Data
약 6개월 간 축구 뉴스 기사 데이터입니다. 총 9077개의 기사 제목과 기사 내용, 기사 발행일 스포츠 뉴스 기사로 이루어져있습니다.
<img width="672" alt="image" src="https://user-images.githubusercontent.com/66352658/159835990-531ab34f-3eef-4493-900b-7a2f97334273.png">
* Title: 기사 제목
* Content : 기사 내용
* Publish_DT : 기사 발행일


## Data Preprocessing
Noise를 제거하기 위해 데이터를 전처리합니다. 9077개로 이루어져있는 원본 데이터를 가공하여 총 9050개의 후처리 데이터를 산출했습니다. 전처리된 데이터는 스포츠 기사 내용, 제목, 기사 발행일 총 세 개의 칼럼으로 이루어져 있습니다.  
모델 성능을 변화시킬 수 있는 데이터 전처리 변인요소로 Stopword, 축약어, 괄호, 날짜를 고려했습니다. 첫 번째로, 뉴스 성능에 영향을 줄 수 있는 Stopword를 직접 지정하여 제거했습니다. 기자 이름, 신문사는 Text Summarization Task에 필요없는 텍스트라고 판단하여 직접 데이터를 리딩하며 Stopword List를 작성하여 제거했습니다. 두번째로,  모델이 축약어의 의미를 이해하여 기사 내 유사한 단어와 연결시켜서 작동할 수 있도록 풀어쓰는 방법을 고안하였습니다. 같은 철자의 축약어가 다른 의미를 가지는 경우와 축약어가 잘못 대체되는 경우 (Coding is good -> Coding 이슬람 근본주의를 표방하는 국제 범죄 단체 good) 위 두 가지를 사람이 모두 살펴가며 검증하기에 로드가 너무 많이 걸리는 이슈를 파악했습니다. 위 단점으로 인해 축약어를 유지하였습니다. 세 번째로, 괄호 안의 값은 전체 텍스트에 중요한 요소가 아니라는 점을 고려하여 괄호 기호와 괄호 사이 내용을 제거했습니다. 네번째로, 기사의 핵심 문장을 추출하는데에 사건이 핵심 문장이 되는 경우가 더 일반적이라고 판단하여 n년/n월/n일, n.n 형태의 날짜 텍스트를 삭제했습니다. 네번째로 중복데이터와 None데이터를 삭제했습니다.

<img width="709" alt="image" src="https://user-images.githubusercontent.com/66352658/159690453-da72473a-69f4-4ee2-a90c-c9db387bbcdf.png">


## Data Augumentation
데이터 과적합을 방지하기 위해 데이터를 증강했습니다. 논문을 서치하여 Back Translation과 EDA 총 두 가지 텍스트 증강 기법을 선정했습니다. 위 기법으로 편향이 적게 발생하도록 하면서 기존 데이터 개수를 늘려 모델의 일반화 성능을 올렸습니다. 최종 데이터는 Train과 Test를 9:1 비율입니다. Test 데이터는 모델의 성능을 평가하기 위한 지표로 사용되는 데이터 셋입니다. 우리 팀은 조금이라도 더 많은 양의 데이터를 학습해서 모델 자체의 성능을 끌어올리는게 중요하다고 생각하여 위와 같은 데이터 비율을 선정했습니다.  
[데이터 비율 선정 참고자료](https://brunch.co.kr/@coolmindory/31#:~:text=%EC%95%84%EB%A7%88%EB%8F%84%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%84%20%EA%B3%B5%EB%B6%80,%EB%A5%BC%20%EB%A7%8E%EC%9D%B4%20%EB%93%A4%EC%96%B4%EB%B3%B4%EC%95%98%EC%9D%84%20%EA%B2%83%EC%9D%B4%EB%8B%A4) 

### Back Translation
Back Translation은 한글 텍스트를 영어 텍스트로 변환한 다음 다시 한 번 한글로 번역하는 방식입니다. Source Sentence를 주어 Target Sentence를 생성하고 이로 변형된 Source Sentence는 방식을 사용하여 자연스러운 인공데이터를 생성합니다. 긴 문장을 Back Translation하는 경우 두 개의 문장으로 나눠지는 경우가 있어서 마침표를 살려서 전처리 했습니다. 데이터를 증강한 후 중복되는 데이터는 제거했습니다.   
  
Back Translation은 어떤 번역기를 선택하느냐에 따라 성능이 좌우된다는 점을 고려했습니다. Pororo, Papago, Googletrans 세 가지 번역 API를 염두해 직접 성능을 실험했습니다. 세 가지 번역기 모두 번역 품질은 우수했습니다. Papago를 사용할 경우 1,000,000 글자 당 20,000 원이 부과된다는 금전적인 이슈,  Googletrans를 사용할 경우 Google Translation 웹 버전의 제한으로 인해 항상 제대로 작동하지 않을 수 있다는 불안정성 이슈로 인해 시간이 더 오래걸린다는 점을 감안하고 Pororo Translator를 선정하여 BackTranslation을 진행했습니다. [실험 코드](https://github.com/seawavve/NLP_wavve/blob/main/Translator_%EB%B9%84%EA%B5%90%EB%B6%84%EC%84%9D.ipynb)
<img width="705" alt="image" src="https://user-images.githubusercontent.com/66352658/159211089-c3f84eec-0ddc-4901-9e12-dc108bbb3d73.png">
    
### EDA (Easy Data Augmentation)
[EDA](https://arxiv.org/pdf/1901.11196v2.pdf)는 4가지 증강 기법을 사용하여 텍스트 데이터의 양을 증강시키는 방법입니다. SR, RI, RS, RD 기법을 사용합니다.
   - SR(Synonym Replacement):문장에서 랜덤으로 불용어를 제외한 n개의 단어를 선택하여 동의어로 바꾼다.  
     봄 날씨 너무 좋지 않니? => 봄 계절 너무 좋지 않니?
   - RI(Random Insertion): 문장에서 랜덤으로 불용어를 제외하여 단어를 선택하고, 해당 단어의 유의어를 문장 내 임의의 자리에 넣는다. 이를 n번 반복한다.  
     봄 날씨 너무 좋지 않니? => 봄 날씨 가을 너무 좋지 않니?
   - RS(Random Swap): 무작위로 문장 내에서 두 단어를 선택하고 위치를 바꾼다. 이를 n번 반복한다.  
     봄 날씨 너무 좋지 않니? => 봄 않니 너무 좋지 날씨?
   - RD(Random Deletion): 단어마다 p의 확률로 랜덤하게 삭제한다.  
     봄 날씨 너무 좋지 않니? => 봄 너무 좋지 않니?

RI와 SR의 경우 해당 언어 Wordmap이 필요한 증강 기법입니다. KAIST에서 배포한 한글 워드맵 Korean Wordnet을 사용하여 두 기법을 테스트 해 본 결과 부적절한 유의어로 증강되는 경우가 있었습니다. 위 이슈를 해결하기 위해 국립국어원에서 제공하는 유의어 사전 자료를 사용했습니다. [국립국어원](https://corpus.korean.go.kr/)에서 어휘 관계 자료 : NIKLex를 토대로 유의어 사전을 만들었습니다. <우리말샘>에 등록된 비슷한말, 반대말, 상위어, 하위어 어휘 쌍을 대상으로 어휘 관계 강도를 5점 척도로 총 5만 명이 평가한 자료입니다. 기존 Korean Wordnet의 경우 유사도를 제공하는 단어 수가 9714개로 적고 각 단어의 유사도의 수준도 점수로 제공되지 않는 반면 모두의 말뭉치 자료에서는 60000개의 단어쌍의 유사도와 점수를 제공합니다. 모두의 말뭉치를 통해 유의어 개수가 적고 신뢰도가 적다는 한국어 Wordnet 문제를 해결했습니다. 유의미한 유의어를 다루기 위해 유의어 단어 쌍 유사도의 전체 평균인 3.7284703333333455 이상인 유의어 단어쌍을 사용했습니다. 최종적으로 모두의 말뭉치 유의어 사전에서 단어간 유사도가 평균 유사도 단어쌍보다 높은 33895개의 한국어 단어 유사도 단어쌍을 사용하여 사전을 만들었습니다.  


아래 표는 논문에 소개된 4가지 기법의 파라미터 설정에 따른 성능 실험 그래프입니다. 알파 파라미터는 각 증강에 의해 변경된 문장의 단어 비율입니다. 이 파라미터 값이 약 0.1~0.2 사이일 때 최적 성능이라는 논문 연구 결과에 따라 본 코드에서는 모든 EDA 기법의 알파 파라미터를 0.2로 설정합니다.

   <img src = "https://user-images.githubusercontent.com/43432539/159114591-ded43c28-5dde-414e-9af9-73b7289d20bd.png" width="800"/>


### True Summary Generation
True Summary(Y 값)은 키워드 기반 랭킹 알고리즘 [BM25](https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html)를 사용해서 Extractive하게 원본데이터에서 핵심문장을 추출 생성했습니다. BM25는 문장 간 유사도 그래프를 기반으로 유사도 측정 함수를 이용해 핵심 문장을 추출합니다. 자연어 처리 오픈소스 라이브러리 Gensim의 summarization 패키지를 사용하여 개발했습니다. 

Y값을 추출하는데 있어서 Gensim/Summarize, TextRank, LexRank, 원본 데이터 타이틀 이 네 가지 Context 대표 문장이나 문장의 대표 단어를 뽑는 Summarization 방식 중에 고민했습니다. TextRank의 경우 요약문의 부자연스러움, 원본 데이터 타이틀 칼럼의 경우 요약 내용 불분명으로 두 방식을 배제했습니다. Gensim/Summarize가 LexRank와 유사한 성능을 내면서 개발 용이성의 장점도 갖고 있었기에 Gensim/Summarize 방식을 선택했습니다.
<img width="656" alt="image" src="https://user-images.githubusercontent.com/66352658/159524062-ea14377a-ce3d-4da6-9fa0-090bec2d2e9b.png">

Gensim/summarize의 BM25는 ratio, word_count 두 개의 파라미터로 결과값을 조정할 수 있습니다. 두 개 파라미터를 적절하게 설정하기 위해 고민했습니다.
첫 번째 파라미터인 ratio는 전체 텍스트에서 요약 텍스트를 뽑을 비율을 결정하는 0과 1 사이의 실수값입니다. 원본 텍스트가 아무리 길더라도 타이틀 문장은 한 문장으로 이루어져있다는 데이터 특성을 고려하여 위 파라미터는 사용하지 않았습니다. 두 번째 파라미터인 word_count는 요약 텍스트 단어 길이 수 입니다. 위와 같이 데이터 특성을 고려한다면 word_count를 사용하는 것이 더 적절하다고 판단했습니다. 필요 이상으로 요약되지않고, 불필요한 내용까지 포함하지도 않는 word_count 값을 고민한 결과 첫 세 문장을 전체 Context의 핵심 문장으로 보는 LEAD-3방식을 착안하여 각 기사의 앞 세 문장의 길이로 선정했습니다. 임의로 정량 숫자를 정하는 것보다 각 기사의 핵심 문장은 앞 세 문장에 위치할 가능성이 크기 때문에 최소한의 데이터 손실을 막도록 앞 세 문장의 길이를 word_count 값으로 선정했습니다.


### Model
한글 스포츠 기사 Text Summarization Task를 해결하기 위해 [seujung/KoBART-summarization](https://github.com/seujung/KoBART-summarization)의 KoBART base 모델을 사용했습니다. 이 모델은 huggingface의 [gogamza/kobart-base-v1](https://huggingface.co/gogamza/kobart-base-v1)입니다. 
* 데이터 적합성  
한글 데이터를 다루기에 한글 언어에 맞춰진 Pretrained Model 사용을 고려했습니다. 
* 개발 용이성  
개발하는 방법이 튜토리얼로 잘 구성되어있는 개발 가능한 모델을 선정했습니다. 본 모델의 경우 Usage를 명시하여 모델 작동을 직관적으로 이해할 수 있습니다.
* 모델 크기  
팀이 현재 사용 가능한 자원에서 충분히 돌아갈 수 있는 모델인지 고려했습니다.

### KoBART
[KoBART(Bidirectional and Auto-Regressive Transformers)](https://arxiv.org/pdf/1910.13461.pdf)는 입력 텍스트 일부에 노이즈를 추가하여 이를 다시 원문으로 복구하는 Autoencoder의 형태로 학습한 Encoder-Decoder 한국어 언어모델입니다. KoBART의 베이스가 되는 BART는 Transformer 기반 언어모델 입니다. BART는 언어 이해에도 좋은 성능을 보이지만 Pretrain된 경우 특히 언어 생성에서 뛰어난 성능을 보여줍니다. 텍스트에 임의의 노이즈를 주어 원본 텍스트를 재구성하는 방식으로 학습합니다. Token Masking, Sentence Permutation, Document Rotation, Token Deletion, Text Infilling 총 다섯가지 함수를 사용하여 BART 모델을 Pretrain합니다. 위 노이즈 함수를 준 텍스트에서 원본 텍스트를 복구하는 방식으로 학습합니다. 다섯가지 Corruption 함수 중에 Text Infilling이 성능이 제일 좋게 나왔고 중요하다는 점을 논문에서 강조합니다.  
  
<img width="463" alt="image" src="https://user-images.githubusercontent.com/66352658/159631341-b1aea072-6973-4bde-8e77-d09084f74080.png">      
  
* Token Masking: 임의로 토큰을 가리는(Masking) 방식
* Sentence Permutation: 문서 내의 문장을 무작위 순서로 재배치
* Document Rotation: 임의의 토큰을 잡아 해당 토큰부터 문서가 시작하도록 회전. 문서의 시작점을 학습
* Token Deletion: 임의로 토큰을 삭제하는 방식. Token Masking과는 다르게 모델이 직접 토큰이 빠진 위치를 유추
* Text Infilling: Poisson distribution (λ = 3)로 여러 텍스트의 span을 샘플링. 각 span은 [Mask] 토큰으로 가려짐


## Param Tuning
어떠한 파라미터가 모델 성능에 어떻게 영향을 미치는지 알아보기 위해 파라미터 튜닝을 실험합니다. 방대한 데이터 양으로 인해 학습에 큰 영향을 미칠 하이퍼 파라미터 조합 2가지만 선정하여 실험 했습니다. 
조절한 파라미터는 batch_size, optimizer_parameters_learning_rate, weight_decay입니다.  

batch size: 실험 환경에서 적용할 수 있는 최대 batch size인 8과 4를 비교하였습니다.

learning rate: learning rate는 1e-5와 5e-5 사이의 값을 선정하기로 하였고, 최소값인 1e-5와 중간값인 5e-5를 선정하여 비교하였습니다.

weight decay: 첫번째 모델의 weight decay는 0.1과 0.01 사이의 균등 분포에서 랜덤하게 추출하였습니다. 그 결과, 0.077이 선정되었습니다.
두번째 모델은 첫 모델의 weight decay의 최대 최소 값과 뽑힌 0.77을 고려하여 0.03으로 선정하여 실험하였습니다.

* 첫번째 실험  
   [hyper-parameter]
   * batch size: 4
   * learning rate: 1e-05
   * weight decay: 0.077
   
   [testset score]
   * ROUGE-L: (F1, Precision, Recall) = (0.527, 0.508, 0.570)
   

![image](https://user-images.githubusercontent.com/61862332/159842960-20c9b794-658f-4204-920e-27ab27035e47.png)


* 두번째 실험  
   [hyper-parameter]
   * batch size: 8
   * learning rate: 3e-05
   * weight decay: 0.03
   
   [testset score]
   * ROUGE-L: (F1, Precision, Recall) = (0.531, 0.510, 0.578)

![image](https://user-images.githubusercontent.com/61862332/159838477-01bfdbd1-8cd1-4617-8323-b6b082336010.png)


### Result
두 실험의 metric은 rouge score를 사용하였습니다. 

[ROUGE SCORE]

ROUGE SCORE는 두 문장 사이에 겹치는 단어가 얼마나 존재하는지를 나타내는 metric입니다.
ROUGE RECALL은 참조 요약본 내 모든 unigram이 모델이 생성한 시스템 요약본에 등장했다는 것을 의미하며 다음의 식을 이용하여 도출됩니다.

![image](https://user-images.githubusercontent.com/61862332/159839191-48b187b1-cfec-4e7b-a345-5fff869522c0.png)

ROUGE PRECISION은 Recall과 반대로 모델이 생성한 시스템 요약본 중 참조 요약본과 겹치는 단어들이 얼마나 많이 존재하는지를 측정합니다.

![image](https://user-images.githubusercontent.com/61862332/159839267-004073dc-d313-4460-8d84-d59c46e41ae5.png)

위와 같이 도출된 recall과 precision 값을 통해 산술평균하여 F1 score도 도출할 수 있게 됩니다.

ROUGE-N은 unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표이며 ROUGE-L은 LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정합니다.
LCS는 ROUGE-2와 같이 단어들의 연속적 매칭을 요구하지 않고, 어떻게든 문자열 내에서 발생하는 매칭을 측정하기 때문에 보다 유연한 성능 비교가 가능합니다.
따라서 저희는 ROUGE-L metric을 이용하여 모델의 성능을 측정 및 비교하였습니다.

최종적으로, 모든 지표에서 두번째 모델에서 더 좋은 성능을 도출할 수 있었습니다. 따라서 저희는 위 제시된 두번째 모델을 최종 모델로 선정하였고 선정된 모델을 이용한 summarization 결과는 다음과 같습니다.



[summarization 예시]

**context**

스페인 프리메라리가 챔피언 아틀레티코 마드리드가 부활한 제시 린가드를 노린다. 영국 매체 90min 은 린가드가 아틀레티코의 영입 타깃이 됐다 며 웨스트햄 유나이티드 이적이 유력해 보이는 상황에서 아틀레티코가 끼어들어 상황이 달라졌다 고 전했다. 린가드는 성장세가 멈추면서 가까이 골을 넣지 못하는 등 조롱의 대상이었다. 그러나 지난 시즌 후반기 웨스트햄으로 임대 이적해 180도 달라졌다. 웨스트햄에서 꾸준히 출전 시간을 부여받은 그는 16경기서 9골을 터뜨리면서 유로파리그 진출을 이끌었다. 단숨에 임대 신화를 쓴 그는 웨스트햄으로 완전 이적할 것으로 보였다. 데이비드 모예스 감독을 비롯해 웨스트햄 수뇌부도 린가드를 데려올 계획을 세웠다. 관건은 이적료다. 맨유는 린가드의 부활에도 판매 의향을 드러냈다. 또 다른 매체 디 애슬레틱 에 따르면 맨유는 린가드의 이적료로 3천만 파운드를 책정했다. 웨스트햄은 부담스러운 반응을 보였고 아틀레티코가 파고들었다. 디에고 시메오네 감독은 활동량이 풍부한 선수들로 수비적인 운영을 주로하기에 일단 많이 뛰는 린가드를 높이 평가할 수 있다. 

**summarization**

영국 매체 90min 은 린가드가 아틀레티코의 영입 타깃이 됐다 며 웨스트햄 유나이티드 이적이 유력해 보이는 상황에서 아틀레티코가 끼어들어 상황이 달라졌다 고 전했다. 그러나 지난 시즌 후반기 웨스트햄으로 임대 이적해 180도 달라졌다. 데이비드 모예스 감독을 비롯해 웨스트햄 수뇌부도 린가드를 데려올 계획을 세웠다.

# References

### Paper
- [BART 논문 정리 블로그](https://jiwunghyun.medium.com/acl-2020-bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-7a0ae37109dc)
- [summarization 모델 분석 비교:사전학습 기반의 법률문서 요약 방법 비교연구, 2021](https://www.koreascience.or.kr/article/CFKO202133648804918.pdf)
- [BART 원 논문](https://arxiv.org/pdf/1910.13461.pdf)

### model
- [KoBART](https://github.com/seujung/KoBART-summarization/tree/4a885fb88f070068197afee692a89fbdbf99345d)
- [KoBART-news](https://huggingface.co/ainize/kobart-news/tree/main)
- [BART metric](https://www.koreascience.or.kr/article/JAKO202024852036141.pdf)
- [ROUGE score](https://huffon.github.io/2019/12/07/rouge/)

### BM25

### Textrank
- [Textrank](https://lovit.github.io/nlp/2019/04/30/textrank/)

## Framework

### Wandb

- [Pytorch lightning + Wandb](https://wandb.ai/wandb_fc/korean/reports/Weights-Biases-Pytorch-Lightning---VmlldzozNzAxOTg)
- [pytorch lightning + Wandb colab](https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW)
- [hyperparameter 값 설정 관련](https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU)
- [early stopping 관련 hyperband](https://homes.cs.washington.edu/~jamieson/hyperband.html)


## Code Contributors

<p>
<a href="https://github.com/jiho-kang" target="_blank">
  <img x="5" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/43432539?v=4"/>
</a>
<a href="https://github.com/tjddn5242" target="_blank">
  <img x="74" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/61862332?v=4"/>
</a>
<a href="https://github.com/rukim001" target="_blank">
  <img x="143" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/92706101?v=4"/>
</a>
<a href="https://github.com/sw6820" target="_blank">
  <img x="212" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/52646313?v=4"/>
</a>
<a href="https://github.com/yjinheon" target="_blank">
  <img x="281" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/37974827?v=4"/>
</a>
<a href="https://github.com/seawavve" target="_blank">
  <img x="350" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/66352658?v=4"/>
</a>

</p>

