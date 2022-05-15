# Directly related
- [OPTIMUS Organizing Sentences via Pre-trained Modeling of a Latent Space](https://arxiv.org/abs/2004.04092)
    - Combines BERT with GPT-2
- [GLSR-VAE Geodesic Latent Space Regularization for Variational AutoEncoder Architectures](https://arxiv.org/abs/1707.04588)
    - CONTINUOUS-VALUED attribute
    - Goal: Giving structure to the latent space
    - Content:
        - Geodesic latent space allows for "linear" interpolation
        - Fix geometry for latent space
- Pati et al (2020) AR-VAE [Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders](https://arxiv.org/abs/2004.05485)
    - Goal: explicitly encode different continuous-valued attributes along cetain latent dimension (proof on image and music)
    - Content:
        - Force specific attribute to have monotonic relationship with latent code of specific dimension
    - Introduction references
        - the topics of image gender swap, speaking style change
        - **other disentangle approaches**
- [Fader Networks:Manipulating Images by Sliding Attributes](https://proceedings.neurips.cc/paper/2017/hash/3fd60983292458bf7dee75f12d5e9e05-Abstract.html)
    - [ArXiv](https://arxiv.org/abs/1706.00409)
    - work with binary attributes. not useful as we want interpolation (is this really true??)
    - [Different domain usage (kinda, i.e. still image) of fader network](https://arxiv.org/abs/2010.07233)


# Indirectly related
- Bowman et al. (2015) [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    - rnn-based VAE for sentence generation
    - interpolation between known sentences
- Martin et al (2019) [Controllable Sentence Simplification](https://arxiv.org/abs/1910.02677)
    - not disentanglement
    - might be benificial as comparison, if english
    - Content:
        - proposes systems with conditioning attributes, such as length, amount of paraphrasing, etc.
        - modifies Seq2Seq
        
# General information
- Ethayarajh (2019) [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://arxiv.org/abs/1909.00512)
    - inspects context of embeddings in BERT, ELMo and GPT-2
    - Major findings:
        - in all layers of the LM the word representations are found to be anisotropic (despice isotropy has been proven to be benificial)
        - upper layers produce more context-specific representations
        - context-specificity manifests differently in the models
        - most likely the representations to not correspond to a finite number of word-sense representations
    - Gives summary of BERT, ELMo, GPT-2


# Further read up
- [Guided Variational Autoencoder for Disentanglement Learning](https://arxiv.org/abs/2004.01255)
    - general disentanglement, supervised and unsupervised
    - use excitation process (as in electric generators) to increase corresponce between latent variable and attribute label
- [Local Disentanglement in Variational Auto-Encoders Using Jacobian $L_1$ Regularization](https://papers.nips.cc/paper/2021/hash/bfd2308e9e75263970f8079115edebbd-Abstract.html)
    - content unclear as of now
- [Disentangling Disentanglement in Variational Autoencoders](https://proceedings.mlr.press/v97/mathieu19a.html)
    - might yield general decomposition scheme
- [German’s Next Language Model](https://aclanthology.org/2020.coling-main.598/)
    - German BERT Model
- [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/abs/1609.07317)
    - 
- [Disentangled Representation Learning for Non-Parallel Text Style Transfer](https://arxiv.org/abs/1808.04339)
    - Goal: disentangle style and content in VAEs
    - yields comparison metrics
    - Content:
        - use multi-task and adversarial loss
        - non-parallel corpus
        - binary sentiment tag -> might be transferable to simplification
- [Disentangling the Latent Space of (Variational) Autoencoders for NLP](https://link.springer.com/chapter/10.1007/978-3-319-97982-3_13)
    - 
- Bao et al (2019) [Generating Sentences from Disentangled Syntactic and Semantic Spaces](https://arxiv.org/abs/1907.05789)
    - Goal: split both aforementioned spaces
    - Content:
        - split latent space into z_semantic and z_syntactic
        - may sample from syntactic, but infer in semantic space for sentence with same meaning but different syntax
        - use multi-task loss, adversarial loss, adversarial recontstruction loss
        - non-parallel
- [Improved Variational Autoencoders for Text Modeling using Dilated]()
    - 
- [Latent Space Regularization for Explicit Control of Musical Attributes]()
    - 
- [Style Tokens Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis]()
    - 
- [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation]()
    - 
- [Surrogate Gradient Field for Latent Space Manipulation]()
    - 
- [Text Style Transfer via Learning Style Instance Supported Latent Space](https://www.ijcai.org/Proceedings/2020/0526)
    -  