# Directly related
- Kingma et al (2014) - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
  - VAE introduction
- Pati et al (2020) AR-VAE [Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders](https://arxiv.org/abs/2004.05485)
  - will be used as extension for the optimization goal for the VAE of OPTIMUS
  - Goal: explicitly encode different continuous-valued attributes along cetain latent dimension (proof on image and music)
  - Content:
    - Force specific attribute to have monotonic relationship with latent code of specific dimension
  - Introduction references
    - the topics of image gender swap, speaking style change
    - **other disentangle approaches**
- Fu et al (2019) - [Cyclical Annealing Schedule](http://arxiv.org/abs/1903.10145)
  - KL annealing schedule

# Models
## VAE
- Higgins et al (2017) - [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
  - introduce beta-VAE
- Cho et al (2014) - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
  - RNN-based AE (R-AE), used by John et al 2019
- Wang et al (2019) - [Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation](https://proceedings.neurips.cc/paper/2019/hash/8804f94e16ba5b680e239a554a08f7d2-Abstract.html)
  - Transformer AE (T-AE)
  - based on Vaswani et al (2017) - [Attention is all you need](https://arxiv.org/abs/1706.03762) (Transformer Paper)
  - fast-gradient-iterative-modification to edit latent representation towards desired attribute by using a separate classifier
- Nangi et al (2021) - [Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer](https://aclanthology.org/2021.acl-short.7)
  - propose counterfactual-based approach to control the latent space for style transfer
  - no new disentanglement method
  - Based on John et al (2019) general structure. Also use RNN
  - Introduce T-VAE based on T-AE by Wang et al (2019) in appendix


## Optimus
- Li et al (2020) - [OPTIMUS Organizing Sentences via Pre-trained Modeling of a Latent Space](https://arxiv.org/abs/2004.04092)
  - Combines BERT with GPT-2
  - network architecture will be used
- [GPT-2](https://github.com/openai/gpt-2) and accompanying paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - GPT-2 Github page
- Radford et al (2016) - [Neural machine translation of rare words with subword units](https://arxiv.org/abs/1508.07909)
  - **Byte Pair Encoding**
- Li et al (2021) - [Prefix-Tuning Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
  - Allow for tuning on pre-trained OPTIMUS
  - 407 Citations according to Google Scholar

## Evaluation metrics
- Adel et al (2018) - [Discovering Interpretable Representations for Both Deep Generative and Discriminative Models](http://proceedings.mlr.press/v80/adel18a.html)
  - Interpretability score
- Chen et al (2018) - [Isolating Sources of Disentanglement in Variational Autoencoders](https://proceedings.neurips.cc/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html)
  - Mutual Information Gap
- Kumar et al (2018) - [Variational Inference of Disentangled Latent Concepts from Unlabeled Observations](https://arxiv.org/abs/1711.00848)
  - Separated Attribute Predictability Score
- Spearman (1904) - [The Proof and Measurement of Association between Two Things](https://www.jstor.org/stable/1412159?origin=JSTOR-pdf#metadata_info_tab_contents)
  - Spearman Rank
- Eastwood et al (2018) - [A framework for the quantitative evaluation of disentangled representations](https://proceedings.neurips.cc/paper/2017/hash/9cb9ed4f35cf7c2f295cc2bc6f732a84-Abstract.html)
  - several metrics
- Carbonneau et al (2022) - [Measuring Disentanglement: A Review of Metrics](https://arxiv.org/abs/2012.09276)
  - review

## German simplification (transformer)
- [Research group on Language Technology for Accessibility at the university of zurich](https://www.cl.uzh.ch/en/texttechnologies/research/accessibility.html)
  - most of the related papers we found so far stem from this group
- Säuberli et al (2020) - [Benchmarking Data-driven Automatic Text Simplification for German](https://www.zora.uzh.ch/id/eprint/188569/)
  - trains autoencoders on simple German corpi
  - same alignment methods as Battisti et al (2020)
  - ideas for automated text simplification
  - Introduction of the Austria Press Agency (APA) corpus
  - Metrics
    - BLEU
    - SARI
- Mallinson et al (2020) - [Zero-Shot Crosslingual Sentence Simplification](Zero-Shot Crosslingual Sentence Simplification)
    -
- Spring et al (2021) - [Exploring German Multi-Level Text Simplification](https://www.zora.uzh.ch/id/eprint/209615/)
  - Simplification into levels A1, A2, B1
  - uses APA and capito corpus

### Not transformer model
- Suter et al (2016) - [Rule-based Automatic Text Simplification for German](https://www.zora.uzh.ch/id/eprint/128601/1/suter-et-al-2016.pdf)
  - as title says introduces several rules for automatic simplification

## Datasets
### German
- Klaper et al (2013) - [Building a German/Simple German Parallel Corpus for Automatic Text Simplification](https://www.zora.uzh.ch/id/eprint/78610/)
  - Build a simple German corpus by crawling websites
  - Corpus not available online
- Battisti et al. (2020) - [A Corpus for Automatic Readability Assessment and Text Simplification of German](https://www.zora.uzh.ch/id/eprint/192839/)
  - Extends Klaper et al (2013) dataset
- Säuberli et al (2020) - [Benchmarking Data-driven Automatic Text Simplification for German](https://www.zora.uzh.ch/id/eprint/188569/)
  - Introduction of the Austria Press Agency (APA) corpus
  - Standard to level B1 German
  - 3,616 sentence pairs
  - see above
- Rios et al (2021) - [A New Dataset and Efficient Baselines for Document-level Text Simplification in German](https://www.zora.uzh.ch/id/eprint/209007/)
  - Presents three datasets
  - self-collected from 20min.ch
  - Austrian Press Agency corpus by Säuberli et al. (2020)
  - a corpus from [capito](https://www.capito.eu/), no reference given on how to obtain
- Aumiller et al (2022) - [Klexikon: A German Dataset for Joint Summarization and Simplification](https://arxiv.org/abs/2201.07198)
  - present a new german dataset

### English
- Coster et al (2011) - [Simple English Wikipedia: A New Text Simplification Task](https://www.researchgate.net/publication/220874254_Simple_English_Wikipedia_A_New_Text_Simplification_Task)
  - As title says it built a corpus from the simple English Wikipedia
  - Referenced by many

# Latent space disentanglement
## Text (also style transfer)
- John et al (2019) - [Disentangled Representation Learning for Non-Parallel Text Style Transfer](https://aclanthology.org/P19-1041)
  - uses "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
  - Goal: disentangle style and content in VAEs
  - introduces R-VAE
    - uses bag-of-words for content. This might indicate bad accuracy if applied to task at hand
  - Split latent into style and content space
  - yields comparison metrics
  - Content:
    - use multi-task and adversarial loss
    - non-parallel corpus
    - binary sentiment tag -> might be transferable to simplification
- Bao et al (2019) - [Generating Sentences from Disentangled Syntactic and Semantic Spaces](https://arxiv.org/abs/1907.05789)
  - Goal: split latent space into syntactic and semantic space
  - Expand John et al (2019)
  - Content:
    - split latent space into z_semantic and z_syntactic
    - may sample from syntactic, but infer in semantic space for sentence with same meaning but different syntax
    - use multi-task loss, adversarial loss, adversarial recontstruction loss
    - non-parallel
- Cheng et al (2020) - [Improving Disentangled Text Representation Learning with Information-Theoretic Guidance](https://arxiv.org/abs/2006.00693)
  - Use Mutual Information
  - Expand John et al (2019)
- Li et al (2022) - [Variational Autoencoder with Disentanglement Priors for Low-Resource](https://arxiv.org/abs/2202.13363)
  - split latent space into content and task-specific one encoding the labels
  - use mixture Gaussian for content, Gaussian prior for each label
  - Use OPTIMUS
- [Formality Style Transfer with Hybrid Textual Annotations](http://arxiv.org/abs/1903.06353)
  - Style transfer

## Other / Non-text
- Hadjeres et al (2017) - [GLSR-VAE Geodesic Latent Space Regularization for Variational AutoEncoder Architectures](https://arxiv.org/abs/1707.04588)
  - CONTINUOUS-VALUED attribute
  - Goal: Giving structure to the latent space
  - Content:
    - Geodesic latent space allows for "linear" interpolation
    - Fix geometry for latent space
- Lample et al (2017) - [Fader Networks:Manipulating Images by Sliding Attributes](https://proceedings.neurips.cc/paper/2017/hash/3fd60983292458bf7dee75f12d5e9e05-Abstract.html)
  - [ArXiv](https://arxiv.org/abs/1706.00409)
  - work with binary attributes. not useful as we want interpolation (this might be transferable though)
  - [Different domain usage (kinda, i.e. still image) of fader network](https://arxiv.org/abs/2010.07233)

# Misc
- Devlin et al (2019) - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Introduction of BERT
- Ethayarajh et al (2019) - [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://arxiv.org/abs/1909.00512)
  - inspects context of embeddings in BERT, ELMo and GPT-2
  - Major findings:
    - in all layers of the LM the word representations are found to be anisotropic (despice isotropy has been proven to be benificial)
    - upper layers produce more context-specific representations
    - context-specificity manifests differently in the models
    - most likely the representations to not correspond to a finite number of word-sense representations
  - Gives summary of BERT, ELMo, GPT-2
- Bowman et al. (2015) - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
  - rnn-based VAE for sentence generation
  - interpolation between known sentences
- Martin et al (2019) - [Controllable Sentence Simplification](https://arxiv.org/abs/1910.02677)
  - not disentanglement
  - might be benificial as comparison, if english
  - Content:
    - proposes systems with conditioning attributes, such as length, amount of paraphrasing, etc.
    - modifies Seq2Seq
- Guo et al (2018) - [Dynamic Multi-Level Multi-Task Learning for Sentence Simplification](https://arxiv.org/abs/1806.07304)
  - sequence-to-sequence sentence simplification model with pointer-copy mechanism
  - give citations on language simplification. These might be useful for writing task
- Cao et al (2020) - [Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen](https://arxiv.org/abs/2005.00701)
  - Medical dataset
  - present comparable methods
  - Content:
    - relate their task to both simplification as well as style transfer
- Chan et al (2020) - [German's Next Language Model](https://aclanthology.org/2020.coling-main.598/)
  - introduces German BERT and ELECTRA models
  - this will be used in the beginning for sentence encoding
  - this might be used for OPTIMUS
- Pati et al (2019) - [Latent Space Regularization for Explicit Control of Musical Attributes](https://www.researchgate.net/publication/334162542_Latent_Space_Regularization_for_Explicit_Control_of_Musical_Attributes)
  - short version of AR-VAE paper

# Further read up
- Mathieu et al (2019) - [Disentangling Disentanglement in Variational Autoencoders](https://proceedings.mlr.press/v97/mathieu19a.html)
  - might yield general decomposition scheme
- Wang et al (2018) - [Style Tokens Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis]()
  - not clear yet
- [Gradient Estimation Using Stochastic Computation Graphs](http://arxiv.org/abs/1506.05254)
  - not clear yet
- [Layer Normalization](http://arxiv.org/abs/1607.06450)
  - Quite old, might not yield anything
- [Learning Disentangled Representations with Semi-Supervised Deep Generative Models](https://proceedings.neurips.cc/paper/2017/hash/9cb9ed4f35cf7c2f295cc2bc6f732a84-Abstract.html)
  - unsure
- [Mask and Infill: Applying Masked Language Model for Sentiment Transfer](https://www.ijcai.org/proceedings/2019/732)
  - Sentiment transfer
- [Non-Parallel Text Style Transfer using Self-Attentional Discriminator as Supervisor](https://ieeexplore.ieee.org/abstract/document/9671820/)
  - style transfer
- [Posterior Collapse and Latent Variable Non-identifiability](https://proceedings.neurips.cc/paper/2021/hash/2b6921f2c64dee16ba21ebf17f3c2c92-Abstract.html)
  - Post collapse
- [Regularization via Structural Label Smoothing](https://proceedings.mlr.press/v108/li20e.html)
  - unsure
- [Training Tips for the Transformer Model](http://content.sciendo.com/view/journals/pralin/110/1/article-p43.xml)
  - misc
- [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation](http://arxiv.org/abs/2101.00828)
  - unsure
- [“Transforming” Delete, Retrieve, Generate Approach for Controlled Text Style Transfer](https://aclanthology.org/D19-1322)
  - style transfer
- [Tutorial on Variational Autoencoders](http://arxiv.org/abs/1606.05908)
  - titles says it all
- [Understanding Posterior Collapse in Generative Latent Variable Models](https://openreview.net/forum?id=r1xaVLUYuE)
  - Post collapse explanation
- [Using the Output Embedding to Improve Language Models](http://arxiv.org/abs/1608.05859)
  - not needed?
- [Z-Forcing: Training Stochastic Recurrent Networks](http://arxiv.org/abs/1711.05411)
  - word sampling

# On hold
## Disentanglement
- Ding (2020) - [Guided Variational Autoencoder for Disentanglement Learning](https://arxiv.org/abs/2004.01255)
  - Guided-VAE for as a controllable generative model
  - work on images
  - Content:
    - general disentanglement, supervised and unsupervised
    - use excitation process (as in electric generators) to increase corresponce between latent variable and attribute label
- Rhodes et al (2021) - [Local Disentanglement in Variational Auto-Encoders Using Jacobian $L_1$ Regularization](https://papers.nips.cc/paper/2021/hash/bfd2308e9e75263970f8079115edebbd-Abstract.html)
  - on images

## Other
- Miao et al (2016) - [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/abs/1609.07317)
  - VAE trained for compressing sentences
  - seems outdated
- Devaraj et al (2022) - [Evaluating Factuality in Text Simplification](https://aclanthology.org/2022.acl-long.506/)
  - Text simplification
- Yang et al (2017) - [Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](http://proceedings.mlr.press/v70/yang17d.html)
  - apply dilated convolution to VAE
  - lack of relevance and seems a bit outdated
- Brunner et al (2018) - [Disentangling the Latent Space of (Variational) Autoencoders for NLP](https://link.springer.com/chapter/10.1007/978-3-319-97982-3_13)
  - no real relevant findings?
- Bostrom et al (2020) - [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720)
  - BPE might be restrictive

## Not related
- Dai et al (2019) - [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation]()
  - No disentanglement, no VAE
- Li et al (2021) - [Surrogate Gradient Field for Latent Space Manipulation]()
  - Images and not using VAE
- Yi et al (2020) - [Text Style Transfer via Learning Style Instance Supported Latent Space](https://www.ijcai.org/Proceedings/2020/0526)
  - Don't use VAE
