# Directly related
- [OPTIMUS Organizing Sentences via Pre-trained Modeling of a Latent Space](https://arxiv.org/abs/2004.04092)
    - Combines BERT with GPT-2
    - network architecture will be used
- Pati et al (2020) AR-VAE [Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders](https://arxiv.org/abs/2004.05485)
    - will be used as extension for the optimization goal for the VAE of OPTIMUS
    - Goal: explicitly encode different continuous-valued attributes along cetain latent dimension (proof on image and music)
    - Content:
        - Force specific attribute to have monotonic relationship with latent code of specific dimension
    - Introduction references
        - the topics of image gender swap, speaking style change
        - **other disentangle approaches**
- [German’s Next Language Model](https://aclanthology.org/2020.coling-main.598/)
    - introduces German BERT and ELECTRA models
    - this might be used for OPTIMUS


## German simplification (transformer)
- [Research group on Language Technology for Accessibility at the university of zurich](https://www.cl.uzh.ch/en/texttechnologies/research/accessibility.html)
    - most of the related papers we found so far stem from this group
- Säuberli et al (2020) [Benchmarking Data-driven Automatic Text Simplification for German](https://www.zora.uzh.ch/id/eprint/188569/)
    - trains autoencoders on simple German corpi
    - same alignment methods as Battisti et al (2020)
    - ideas for automated text simplification
    - Introduction of the Austria Press Agency (APA) corpus
    - Metrics
        - BLEU
        - SARI
- Mallinson et al (2020) [Zero-Shot Crosslingual Sentence Simplification](Zero-Shot Crosslingual Sentence Simplification
)
    - 
- Spring et al (2021) [Exploring German Multi-Level Text Simplification](https://www.zora.uzh.ch/id/eprint/209615/)
    - Simplification into levels A1, A2, B1
    - uses APA and capito corpus

### Not transformer model
- Suter et al (2016) [Rule-based Automatic Text Simplification for German](https://www.zora.uzh.ch/id/eprint/128601/1/suter-et-al-2016.pdf)
    - as title says introduces several rules for automatic simplification

## Datasets
### German
- Klaper et al (2013) [Building a German/Simple German Parallel Corpus for Automatic Text Simplification](https://www.zora.uzh.ch/id/eprint/78610/)
    - Build a simple German corpus by crawling websites
    - Corpus not available online
- Battisti et al. (2020) [A Corpus for Automatic Readability Assessment and Text Simplification of German](https://www.zora.uzh.ch/id/eprint/192839/)
    - Extends Klaper et al (2013) dataset
- Säuberli et al (2020) [Benchmarking Data-driven Automatic Text Simplification for German](https://www.zora.uzh.ch/id/eprint/188569/)
    - Introduction of the Austria Press Agency (APA) corpus
    - Standard to level B1 German
    - 3,616 sentence pairs
    - see above
- Rios et al (2021) [A New Dataset and Efficient Baselines for Document-level Text Simplification in German](https://www.zora.uzh.ch/id/eprint/209007/)
    - Presents three datasets
    - self-collected from 20min.ch
    - Austrian Press Agency corpus by Säuberli et al. (2020)
    - a corpus from [capito](https://www.capito.eu/), no reference given on how to obtain
- Aumiller et al (2022) [Klexikon: A German Dataset for Joint Summarization and Simplification](https://arxiv.org/abs/2201.07198)
    - present a new german dataset

### English
- Coster et al (2011) [Simple English Wikipedia: A New Text Simplification Task](https://www.researchgate.net/publication/220874254_Simple_English_Wikipedia_A_New_Text_Simplification_Task)
    - As title says it built a corpus from the simple English Wikipedia
    - Referenced by many




# Latent space disentanglement
## Text
- John et al (2018) [Disentangled Representation Learning for Non-Parallel Text Style Transfer](https://arxiv.org/abs/1808.04339)
    - Goal: disentangle style and content in VAEs
    - yields comparison metrics
    - Content:
        - use multi-task and adversarial loss
        - non-parallel corpus
        - binary sentiment tag -> might be transferable to simplification
- Bao et al (2019) [Generating Sentences from Disentangled Syntactic and Semantic Spaces](https://arxiv.org/abs/1907.05789)
    - Goal: split latent space into syntactic and semantic space
    - Content:
        - split latent space into z_semantic and z_syntactic
        - may sample from syntactic, but infer in semantic space for sentence with same meaning but different syntax
        - use multi-task loss, adversarial loss, adversarial recontstruction loss
        - non-parallel
## Other
- Hadjeres et al (2017) [GLSR-VAE Geodesic Latent Space Regularization for Variational AutoEncoder Architectures](https://arxiv.org/abs/1707.04588)
    - CONTINUOUS-VALUED attribute
    - Goal: Giving structure to the latent space
    - Content:
        - Geodesic latent space allows for "linear" interpolation
        - Fix geometry for latent space
- [Fader Networks:Manipulating Images by Sliding Attributes](https://proceedings.neurips.cc/paper/2017/hash/3fd60983292458bf7dee75f12d5e9e05-Abstract.html)
    - [ArXiv](https://arxiv.org/abs/1706.00409)
    - work with binary attributes. not useful as we want interpolation (this might be transferable though)
    - [Different domain usage (kinda, i.e. still image) of fader network](https://arxiv.org/abs/2010.07233)



# General information
- Ethayarajh (2019) [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://arxiv.org/abs/1909.00512)
    - inspects context of embeddings in BERT, ELMo and GPT-2
    - Major findings:
        - in all layers of the LM the word representations are found to be anisotropic (despice isotropy has been proven to be benificial)
        - upper layers produce more context-specific representations
        - context-specificity manifests differently in the models
        - most likely the representations to not correspond to a finite number of word-sense representations
    - Gives summary of BERT, ELMo, GPT-2



# Misc
- [Measuring Disentanglement: A Review of Metrics](https://arxiv.org/abs/2012.09276)
    - 
- Bowman et al. (2015) [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    - rnn-based VAE for sentence generation
    - interpolation between known sentences
- Martin et al (2019) [Controllable Sentence Simplification](https://arxiv.org/abs/1910.02677)
    - not disentanglement
    - might be benificial as comparison, if english
    - Content:
        - proposes systems with conditioning attributes, such as length, amount of paraphrasing, etc.
        - modifies Seq2Seq
- Pati et al (2019) [Latent Space Regularization for Explicit Control of Musical Attributes](https://www.researchgate.net/publication/334162542_Latent_Space_Regularization_for_Explicit_Control_of_Musical_Attributes)
    - short version of AR-VAE paper

# Further read up
- (2020) [Guided Variational Autoencoder for Disentanglement Learning](https://arxiv.org/abs/2004.01255)
    - Guided-VAE for as a controllable generative model
    - Content:
        - general disentanglement, supervised and unsupervised
        - use excitation process (as in electric generators) to increase corresponce between latent variable and attribute label
- [Local Disentanglement in Variational Auto-Encoders Using Jacobian $L_1$ Regularization](https://papers.nips.cc/paper/2021/hash/bfd2308e9e75263970f8079115edebbd-Abstract.html)
    - content unclear as of now
- [Disentangling Disentanglement in Variational Autoencoders](https://proceedings.mlr.press/v97/mathieu19a.html)
    - might yield general decomposition scheme
- [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/abs/1609.07317)
    - 
- [Style Tokens Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis]()
    - 
- [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation]()
    - 
- [Surrogate Gradient Field for Latent Space Manipulation]()
    - 
- [Text Style Transfer via Learning Style Instance Supported Latent Space](https://www.ijcai.org/Proceedings/2020/0526)
    -    
- Guo et al (2018) [Dynamic Multi-Level Multi-Task Learning for Sentence Simplification](https://arxiv.org/abs/1806.07304)
    - 
- Cao et al (2020) [Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen](https://arxiv.org/abs/2005.00701)
    - Medical dataset
    - present comparable methods
    - Content:
        - relate their task to both simplification as well as style transfer
- Devaraj et al (2022) [Evaluating Factuality in Text Simplification](https://aclanthology.org/2022.acl-long.506/)
    - 

## On hold
- Yang et al (2017) [Improved Variational Autoencoders for Text Modeling using Dilated](http://proceedings.mlr.press/v70/yang17d.html)
    - lack of relevance and seems a bit outdated
- Brunner et al (2018) [Disentangling the Latent Space of (Variational) Autoencoders for NLP](https://link.springer.com/chapter/10.1007/978-3-319-97982-3_13)
    - no real findings?
