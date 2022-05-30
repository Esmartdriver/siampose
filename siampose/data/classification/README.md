# Generic File Based Classification Dataloader
The goal is to provide a generic way to pre-train on unlabelled data from videos, then evaluate how well the embeddings for a never before seen image patch (for instance, object crop) maps to similar image patches.

The underlying network architecture is the same as SimSiam. The motivation behind using siamese networks is to avoid having to define negative pairs. Instead of a contrastive architecture, we use an "attractive" architecture, which is more likely to work well with video data. 

## Dataset Description
### Training Set
The training set will be split in a fully "unlabelled" training set, and a smaller "labelled" training set, which will be used for nearest neighbor lookup.

The unlabelled dataset contains raw video frames, while the labelled training set contains object crops with their respective categories. 

### Validation Set
The validation set is a set of object crops with their respective categories. Those samples will be used a queries against the training set.  

### Test Set
Real world operation perfomance will be used as a test set. 

# Downstream Tasks
## Unsupervised pre-training
During the unsupervised pre-training, overlapping crops from two consecutive videos frames are extratcted and used as a positive pair. Data augmentation is done on each crop to generate diversity. The underlying assumption is that overlapping crops will contain overlapping objects. 

### Positive pairs
The first strategy is to pick a random point on the first image. Assuming a 96x96 projector input size, we can take a random crop arround a location (x,y) of a bigger size, such as 224x196, then scale it back to 96x96. The size of the second crop should be taken arround the same location (x,y), but with a different size, such as 96x128. Again, the second crop would be resize to the 96x96 input size. This would in fact create a "random resized crop" arround a specific location, create a self-annotated "object" in the scene.  

## Qualitative evaluation.
For qualitative evaluation, the goal is to make sure that queries from the validation dataset will retrieve semantically similar objects in the training dataset. For instance, if a patch containing a "cow" is used as a query, and all/most of the resulting patches in the training sets also contain "cows", then it suggests that sementically similar objects are nearby in embedding space. 

This can also be used a a tool to quickly annotate data in the training set, with a "human" confirming the proposals from the system. Qualitative evaluation does not require any labeled data.

## Few-shot classification.
In order to validate the few-shot classification performance, embeddings will be generated for the validation set object crops. A small "few-shot" labelled training set, which can overlap with the unsupervised dataset, will contain the category for object crops. Embeddings will also be generated for this labelled dataset, and a nearest neighbor lookup in embedding space will be used to find the closest sample. 